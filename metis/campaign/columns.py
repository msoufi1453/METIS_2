"""Column names of a campaign's data table and of the workbook's Experiments sheet."""

ID = 'Sample ID'            # one number per recipe; replicates share it
ROUND = 'Round'             # 0 = data that existed before the campaign (day 0), 1, 2, ... = generated rounds
POINT = 'Data point'        # position of the recipe within its round, from 1
REPLICATE = 'Replicate'     # from 1; every replicate has its own row
PLATE = 'Plate'             # liquid handling only
WELL = 'Well'               # liquid handling only, e.g. 1A = column 1, row A
BOOKKEEPING = (ID, ROUND, POINT, REPLICATE, PLATE, WELL)

LOW = '80% interval low'
HIGH = '80% interval high'
INTERVAL = 0.8


def predicted(target):
    return 'Predicted {}'.format(target)


def prediction_columns(target):
    return [predicted(target), LOW, HIGH]


def reserved(target):
    """Names a variable may not use (compared without regard to case)."""
    return {name.lower() for name in BOOKKEEPING + (LOW, HIGH, predicted(target), target)}
