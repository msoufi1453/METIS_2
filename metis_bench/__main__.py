"""Command line for the benchmark.

    python -m metis_bench oracles                      # build and cache the real-data oracles
    python -m metis_bench calibrate                    # descriptors, suite selection, realism map
    python -m metis_bench run experiments/phase2_surrogates.yaml --jobs 8 [--pilot]
    python -m metis_bench report experiments/phase2_surrogates.yaml [--pilot]
    python -m metis_bench replay experiments/replay_paper.yaml      # paper-style figures for measured-data runs
    python -m metis_bench tasks experiments/phase2_surrogates.yaml --shard 3/40   # list one shard (for clusters)
    python -m metis_bench run experiments/phase2_surrogates.yaml --arms surr_tabpfn  # one arm only (e.g. on a GPU)
"""
import argparse
import json
import pathlib
import sys

from . import experiments


def main(argv=None):
    parser = argparse.ArgumentParser(prog='metis_bench')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('oracles')
    cal = sub.add_parser('calibrate')
    cal.add_argument('--candidates', type=int, default=24, help='candidate specs per subtype')
    cal.add_argument('--per-subtype', type=int, default=5)
    cal.add_argument('--out', default='suite')
    cal.add_argument('--jobs', type=int, default=1)
    summ = sub.add_parser('summary', help='one markdown overview of all experiment reports')
    summ.add_argument('--pilot', action='store_true')
    summ.add_argument('--results', default='results')
    summ.add_argument('--out', default=None)
    dec = sub.add_parser('decide', help='apply the preregistered decision rule to Phase 2 reports')
    dec.add_argument('reports', nargs='+', help='report directories (results/<experiment>/report)')
    dec.add_argument('--pilot', action='store_true')
    dec.add_argument('--out', default=None)
    dec.add_argument('--ignore-significance', action='store_true',
                     help='pilots only: pick the best mean per part even if not significant')
    for name in ('run', 'report', 'tasks', 'tune', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('experiment')
        p.add_argument('--pilot', action='store_true')
        p.add_argument('--out', default='results')
        p.add_argument('--jobs', type=int, default=1)
        p.add_argument('--shard', default=None, help='i/n: run only every n-th task starting at i (0-based)')
        p.add_argument('--arms', default=None, help='comma-separated arm names to run (others skipped)')
        p.add_argument('--skip-arms', default=None, help='comma-separated arm names to leave out')
    args = parser.parse_args(argv)

    if args.command == 'oracles':
        from .oracles import REAL_ORACLES
        for name, build in REAL_ORACLES.items():
            o = build()
            print(json.dumps({k: v for k, v in o.describe().items() if k not in ('gold_params',)}, default=str))
        return 0

    if args.command == 'calibrate':
        from . import calibration as C
        refs = C.real_references()
        print(refs.round(3).to_string())
        screened = C.screen_candidates(n_per_subtype=args.candidates, jobs=args.jobs)
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        screened.to_csv(out / 'screened_candidates.csv', index=False)
        selected = C.select_suite(screened, refs, per_subtype=args.per_subtype)
        table = C.finalize_suite(selected, screened, refs, out, jobs=args.jobs)
        print(table.groupby(['landscape', 'n_dims']).realism.describe().round(3).to_string())
        return 0

    if args.command == 'summary':
        from .summary import write_summary
        out = args.out or ('results/SUMMARY-pilot.md' if args.pilot else 'results/SUMMARY.md')
        print(write_summary(args.results, out, pilot=args.pilot))
        return 0

    if args.command == 'decide':
        from .tuning import decide_winners
        out = args.out or ('artifacts/winners-pilot.json' if args.pilot else 'artifacts/winners.json')
        winners = decide_winners(args.reports, out, pilot=args.pilot,
                                 require_significance=not args.ignore_significance)
        print(json.dumps(winners['interventions'], indent=1))
        print('written to', out)
        return 0

    cfg = experiments.load(args.experiment, pilot=args.pilot)
    if args.command == 'report':
        from .report import write_report
        print(write_report(cfg, pathlib.Path(args.out) / cfg['name']))
        return 0
    if args.command == 'replay':
        from .replay import write_replay
        print(write_replay(cfg, pathlib.Path(args.out) / cfg['name']))
        return 0
    tasks = experiments.tasks(cfg, args.out)
    # Split an experiment across machines by arm, e.g. TabPFN on a GPU laptop and the rest
    # on CPU workers; results land in the same layout, so copying the folders together
    # before `report` is all the merging needed.
    if args.arms:
        keep = set(args.arms.split(','))
        tasks = [t for t in tasks if t[1]['name'] in keep]
    if args.skip_arms:
        drop = set(args.skip_arms.split(','))
        tasks = [t for t in tasks if t[1]['name'] not in drop]
    if args.shard:
        i, n = map(int, args.shard.split('/'))
        tasks = tasks[i::n]
    if args.command == 'tasks':
        for t in tasks:
            print(t[4])
        return 0
    from .campaign import run_tasks
    failures = run_tasks(tasks, jobs=args.jobs)
    for path, error in failures:
        print('FAILED', path, '\n', error, file=sys.stderr)
    if args.command == 'tune' and not failures:
        from .report import collect
        from .tuning import select_winner, write_winner
        runs, _, _ = collect(pathlib.Path(args.out) / cfg['name'])
        winner, table = select_winner(runs)
        print(table.round(4).to_string())
        print('winner:', winner, '->', write_winner(cfg['tuning']['output'], cfg, winner, table))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
