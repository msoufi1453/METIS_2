"""The app's charts (Altair). Colours follow one small system: a single blue ramp for a
measured value (darker = higher), bright green for proposed recipes, grey ink for the rest.
Hovering a mark shows its values; the same values are in the tables next to each chart.
"""
import altair as alt
import numpy as np
import pandas as pd

RAMP = ['#cde2fb', '#3987e5', '#0d366b']        # one hue, light to dark
BLUE = '#2a78d6'
GREEN = '#1fd655'                               # proposed recipes
GREEN_INK = '#0b7a2f'
MUTED = '#898781'
SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'triangle-down', 'triangle-right',
          'triangle-left', 'wedge', 'arrow']
MEASURED, PROPOSED, BEFORE, PATH = 'Measured', 'Proposed', 'Before hill-climbing', 'Path'
MAX_TOOLTIP_VARIABLES = 14


def number(value, digits=4):
    """A number as short text; a dash for a missing one."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '–'
    if isinstance(value, str):
        return value
    return '{:.{}g}'.format(float(value), digits)


def _trails(moves, steps=14):
    """Hill-climbing paths as rows of small dots (a selectable chart must be a single layer, so no line marks)."""
    rows = []
    for k, move in enumerate(moves.itertuples(index=False)):
        for i in range(1, steps):
            share = i / steps
            rows.append({'uid': 'T{}_{}'.format(k, i), 'x': move.x + share * (move.x2 - move.x),
                         'y': move.y + share * (move.y2 - move.y), 'group': PATH, 'label': move.label})
    return pd.DataFrame(rows)


def _unless_measured(proposed, before, path, legend=None):
    """Conditions giving the proposed, before-climbing and path marks their own value of a mark property.

    Vega-Lite draws a legend's symbols with the first condition's value, which would make the
    symbols of the round legend green. ``legend`` puts a condition that never holds in front, so
    the symbols take that value instead."""
    first = [] if legend is None else [{'test': 'false', 'value': legend}]
    return first + [{'test': "datum.group === '{}'".format(group), 'value': value}
                    for group, value in ((PROPOSED, proposed), (BEFORE, before), (PATH, path))]


def pca_map(points, target, variables, explained, moves=None, height=460):
    """The recipes on the PCA map.

    ``points``: one row per recipe with ``uid``, ``x``, ``y``, ``group`` (Measured / Proposed /
    Before hill-climbing), ``round`` (text), ``label``, ``result``, ``measured`` and
    ``predicted`` (texts for the tooltip) and a column per variable. ``variables``:
    ``[(column, label)]``. ``moves``: hill-climbing paths with ``x``, ``y``, ``x2``, ``y2``
    and ``label``. Measured recipes are blue (darker = higher result) with a shape per
    round; proposed ones are bright green circles; ``map_legend`` names the green marks.
    Clicking a mark selects it (selection ``pick``, field ``uid``).
    """
    data = points.copy()
    shown = variables[:MAX_TOOLTIP_VARIABLES]
    tips = [alt.Tooltip('label:N', title='Recipe'), alt.Tooltip('measured:N', title='Measured ' + target),
            alt.Tooltip('predicted:N', title='Predicted (80% interval)')]
    for k, (column, label) in enumerate(shown):
        data['tip{}'.format(k)] = [number(v) for v in data[column]]
        tips.append(alt.Tooltip('tip{}:N'.format(k), title=label))
    if len(variables) > len(shown):
        data['tip_more'] = 'click the point for all {}'.format(len(variables))
        tips.append(alt.Tooltip('tip_more:N', title='More variables'))
    keep = ['uid', 'x', 'y', 'group', 'round', 'label', 'result', 'measured', 'predicted'] + \
        [c for c in data.columns if c.startswith('tip')]
    is_measured = data['group'] == MEASURED
    parts = [data.loc[is_measured, keep]]                       # drawn first, so proposals lie on top
    if moves is not None and len(moves):
        parts.append(_trails(moves))
    parts += [data.loc[data['group'] == BEFORE, keep], data.loc[data['group'] == PROPOSED, keep]]
    data = pd.concat([part for part in parts if len(part)], ignore_index=True)
    floor = float(data.loc[data['group'] == MEASURED, 'result'].min()) if is_measured.any() else 0.0
    data['result'] = data['result'].fillna(floor)               # a valid number: only measured points use it
    data = data.fillna('')
    rounds = list(dict.fromkeys(data.loc[data['group'] == MEASURED, 'round']))
    green = _unless_measured(GREEN, 'transparent', GREEN_INK, legend=MUTED)
    if rounds:
        colour = alt.Color('result:Q', title=target, scale=alt.Scale(range=RAMP), condition=green,
                           legend=alt.Legend(orient='right', gradientLength=120))
        shape = alt.Shape('round:N', title='Measured in', sort=rounds,
                          scale=alt.Scale(domain=rounds, range=[SHAPES[i % len(SHAPES)] for i in range(len(rounds))]),
                          legend=alt.Legend(orient='right', symbolFillColor=MUTED, symbolStrokeColor=MUTED),
                          condition={'test': "datum.group !== '{}'".format(MEASURED), 'value': 'circle'})
    else:
        colour, shape = alt.ColorValue(BLUE, condition=green), alt.ShapeValue('circle')
    pick = alt.selection_point(name='pick', fields=['uid'], on='click', clear='dblclick')
    return alt.Chart(data).mark_point(filled=True, opacity=1).encode(
        x=alt.X('x:Q', title='PC 1 ({:.0%} of the variation between the mapped recipes)'.format(explained[0]),
                scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y('y:Q', title='PC 2 ({:.0%})'.format(explained[1]), scale=alt.Scale(zero=False, padding=20)),
        color=colour,
        shape=shape,
        size=alt.SizeValue(95, condition=_unless_measured(150, 150, 9)),
        stroke=alt.StrokeValue(MUTED, condition=_unless_measured(GREEN_INK, GREEN_INK, GREEN_INK, legend=MUTED)),
        strokeWidth=alt.StrokeWidthValue(0.8, condition=_unless_measured(1.5, 1.5, 0)),
        tooltip=tips).add_params(pick).properties(height=height)


def map_legend(proposed_label, before=False, paths=False):
    """HTML naming the green marks of the map (they are not part of the chart's own legends)."""
    def swatch(fill, size=12):
        return ('<span style="display:inline-block;width:{0}px;height:{0}px;border-radius:50%;background:{1};'
                'border:1.5px solid {2};margin:0 6px 0 14px;vertical-align:middle"></span>').format(size, fill, GREEN_INK)
    items = [swatch(GREEN) + proposed_label]
    if before:
        items.append(swatch('transparent') + 'Before hill-climbing')
    if paths:
        items.append(swatch(GREEN_INK, 3) + 'Path of a climb')
    return '<div style="font-size:0.85rem;margin-left:-14px">{}</div>'.format(''.join(items))


def progress(samples, target, round_column, id_column, height=360):
    """Every recipe's result by round, with the best so far as a line."""
    data = samples[[id_column, round_column, target]].dropna()
    data = pd.DataFrame({'id': data[id_column].to_numpy(), 'round': data[round_column].to_numpy(dtype=int),
                         'value': data[target].to_numpy(dtype=float)})
    spread = (data['id'].to_numpy() * 0.6180339887) % 1.0                 # a fixed jitter per recipe
    data['position'] = data['round'] + (spread - 0.5) * 0.5
    data['series'] = 'One recipe'
    best = data.groupby('round', as_index=False)['value'].max().sort_values('round')
    best['value'] = best['value'].cummax()
    best['position'], best['series'] = best['round'].astype(float), 'Best so far'
    for frame in (data, best):
        frame['text'] = [number(v) for v in frame['value']]
    rounds = sorted(int(r) for r in data['round'].unique())
    x = alt.X('position:Q', title='Round', axis=alt.Axis(values=rounds, format='d', grid=False),
              scale=alt.Scale(domain=[min(rounds) - 0.5, max(rounds) + 0.5]))
    y = alt.Y('value:Q', title=target, scale=alt.Scale(zero=False))
    colour = alt.Color('series:N', title=None, scale=alt.Scale(domain=['One recipe', 'Best so far'],
                                                                 range=[BLUE, MUTED]))
    tip = [alt.Tooltip('series:N', title=' '), alt.Tooltip('round:Q', title='Round'),
           alt.Tooltip('text:N', title=target)]
    dots = alt.Chart(data).mark_circle(size=55, opacity=0.8).encode(x=x, y=y, color=colour, tooltip=tip)
    line = alt.Chart(best).mark_line(strokeWidth=2, interpolate='step-after').encode(x=x, y=y, color=colour)
    marks = alt.Chart(best).mark_point(size=70, filled=True, opacity=1).encode(x=x, y=y, color=colour, tooltip=tip)
    return alt.layer(dots, line, marks).properties(height=height)


def ranking(values, title, top=20):
    """Horizontal bars, largest first. ``values``: a Series (name -> value)."""
    data = pd.DataFrame({'name': [str(i) for i in values.index], 'value': values.to_numpy(dtype=float)}).head(top)
    data['text'] = [number(v, 3) for v in data['value']]
    order = list(data['name'])
    base = alt.Chart(data).encode(
        y=alt.Y('name:N', sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X('value:Q', title=title),
        tooltip=[alt.Tooltip('name:N', title=' '), alt.Tooltip('text:N', title=title)])
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, height={'band': 0.7})
    return bars.properties(height=max(120, 26 * len(data) + 40))
