"""Reading day-by-day ``Results_<day>.csv`` files (METIS v1 API)."""
import os

import pandas as pd


def day_finder(file, file_format='csv'):
    """Find the first notcompleted day

    Parameters
    ----------
    file: 
        for now, it can only be 'Results'
        
    Returns
    -------
    i: int
        the first not completed day
    """
    i = 1
    while True:
        if not os.path.isfile('{}_{}.{}'.format(file, i, file_format)):
            return i
        i += 1


def result_preprocess(day, desired_cols, range=20):
    """Preprocess Results.csv file to get desired columns and rows
        caution: the target column name MUST be 'yield'
        
    Parameters
    ----------
    day: 
        Results_day.csv
    
    desired_cols:
        name of columns that you want from the results file
        
    Returns
    -------
    data_m:
        data in range
    label_m:
        label in range
    data_specials:
        other data
    label_specials:
        other labels
    """
    results = pd.read_csv('Results_{}.csv'.format(day, day))

    # m number pipeline
    data_m = results[desired_cols].iloc[:range, :]
    label_m = results[['yield']].iloc[:range, :]

    # reference, control and specials
    data_specials = results[desired_cols].iloc[range:, :]
    label_specials = results[['yield']].iloc[range:, :]

    return data_m, label_m, data_specials, label_specials
