"""Concentration grids, random designs and concentration-to-volume conversion (METIS v1 API).

These functions keep their v1 signatures and outputs; tests/test_regression_legacy.py
checks them byte for byte against the original ``utils.py``. Two internal changes:

* ``random_combination_generator`` checks for repeats with a set instead of a list scan.
  Membership semantics and the random stream are unchanged, but a 100,000-point pool now
  takes seconds instead of hours.
* ``check_repeat`` is honoured. v1 tested the module-level function ``check_repetitive``
  (always truthy) by mistake, so repeats were always rejected; that is still the default.
"""
from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd


def allowed_output(value, reaction_vol_nl=20000, drop_size_nl=100, verbose=0):
    """Based on high ,low and stock concentrations and droplet size calculate how many combinations is possible

    Parameters
    ----------
    value: tuple
        (low, high, stock concentration)

    Returns
    -------
    calculated_concs:
        a list of possible concentrations

    calculated_vols:
        a list of possible volumes
    """

    if value['Conc_Values']:
        if isinstance(value['Conc_Stock'], Iterable):
            drop_nums = [i * reaction_vol_nl / (drop_size_nl * value['Conc_Stock'][find_stock(value['Conc_Values'], value['Conc_Stock'], i)[0]]) for i in value['Conc_Values']]
            calculated_concs = value['Conc_Values']
        else:
            drop_nums = [i * reaction_vol_nl / (drop_size_nl * value['Conc_Stock']) for i in value['Conc_Values']]
            calculated_concs = value['Conc_Values']

    else:
        drop_nums = list(range(int((value['Conc_Min'] * reaction_vol_nl) / (drop_size_nl * value['Conc_Stock'])),
                               int((value['Conc_Max'] * reaction_vol_nl) / (drop_size_nl * value['Conc_Stock'])) + 1))

        calculated_concs = [drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl for drop_num in drop_nums]

    if verbose:
        print('drops :', drop_nums)
        print('volumes :', [i * drop_size_nl for i in drop_nums])
        print('possible_concentrations :', calculated_concs)
    else:
        return calculated_concs, [i * drop_size_nl for i in drop_nums]


def percentage_possible(data, threshold=40):
    """Based on threshold volume, it calculates how many combinations of all metabolite is possible to make

    Parameters
    ----------
    data: dict
        {'meatbolite name':[possible volumes], ...}

    Returns
    -------
    percentage possible: float
    total: int
        total number of combinations (includes forbidden one)
    """
    lists = list(data.values())

    total = np.prod(np.array([len(i) for i in data.values()]))
    possible = 0

    for items in product(*lists):
        if sum(items) <= threshold:
            possible += 1

    return (possible/total*100), total


def find_stock(conc_values, conc_stocks, this_value):
    """this function find each concentration value belongs to wich stock concentration for metabolites with multiple stocks

    Parameters
    ----------
    conc_values: list
        a list of all possible concentration

    conc_stocks: list
        a list of all stocks concentration

    this_value: float, int
        concentration value that we find to find its stock

    Returns
    -------
    i:
        index of found stock

    out:
        value of found stock
    """
    num = len(conc_stocks)
    avg = len(conc_values) / float(num)
    out = []
    last = 0.0

    while last < len(conc_values):
        out.append(conc_values[int(last):int(last + avg)])
        last += avg

    for i, value in enumerate(out):
        if this_value in value:
            return i, out


# random combination generator function_v3.0
def random_combination_generator(concentrations_limits, number_of_combination=100, reaction_vol_nl=10000,
                                 max_nl=None, drop_size_nl=100, check_repeat=True, rounded=2, verbose=0, make_csv=False, return_df=False):
    """this function make random combination that is safe (e.g. dont make too much or low concentrated, not excecutable based on drop size, not repetitive)

    Parameters
    ----------
    concentrations_limits: dict
        {'name of metabolite': {'Conc_Min': #, 'Conc_Max': #, 'Conc_Values': #, 'Conc_Stock': #, 'Alternatives': #}, ...}

    Returns
    -------
    data: pandas.DataFrame
        a dataframe as consists of number_of_combination of random combinations
    """

    # generating random combinations
    combinations = []
    seen = set()
    data_point = 0
    while data_point < number_of_combination:
        input_data = []
        input_vol = []
        # verbosity
        if (data_point % 10000 == 0) and verbose:
            print(data_point)

        # generation of random input
        for key, value in concentrations_limits.items():
            # Manual Concentration Value Generation
            if value['Conc_Values']:
                # With Alternatives
                if value['Alternatives']:
                    num_alternative = len(value['Alternatives'])
                    choice_alternative = np.random.randint(0, num_alternative)
                    choice_list = [0 for i in range(num_alternative)]
                    choice_list[choice_alternative] = 1

                    choice_conc = np.random.choice(value['Conc_Values'])
                    input_data.append(choice_conc)
                    input_data += choice_list
                    if isinstance(value['Conc_Stock'], Iterable):
                        choice_stock, _ = find_stock(value['Conc_Values'], value['Conc_Stock'], choice_conc)
                        input_vol.append(choice_conc/value['Conc_Stock'][choice_stock]*reaction_vol_nl)
                    else:
                        input_vol.append(choice_conc/value['Conc_Stock']*reaction_vol_nl)

                # Without Alternatives
                else:
                    choice_conc = np.random.choice(value['Conc_Values'])
                    input_data.append(choice_conc)
                    if isinstance(value['Conc_Stock'], Iterable):
                        choice_stock, _ = find_stock(value['Conc_Values'], value['Conc_Stock'], choice_conc)
                        input_vol.append(choice_conc/value['Conc_Stock'][choice_stock]*reaction_vol_nl)
                    else:
                        input_vol.append(choice_conc/value['Conc_Stock']*reaction_vol_nl)

            # Auto Concentration Value Generation
            else:
                # With Alternatives
                if value['Alternatives']:
                    num_alternative = len(value['Alternatives'])
                    choice_alternative = np.random.randint(0, num_alternative)
                    choice_list = [0 for i in range(num_alternative)]
                    choice_list[choice_alternative] = 1

                    drop_num = np.random.randint(round(value['Conc_Min'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']),
                                                 round(value['Conc_Max'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']) + 1)

                    recalculated_conc = drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl
                    input_data.append(recalculated_conc)
                    input_data += choice_list
                    input_vol.append(recalculated_conc/value['Conc_Stock']*reaction_vol_nl)

                # Without Alternatives
                else:
                    drop_num = np.random.randint(round(value['Conc_Min'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']),
                                                 round(value['Conc_Max'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']) + 1)

                    recalculated_conc = drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl
                    input_data.append(recalculated_conc)
                    input_vol.append(recalculated_conc/value['Conc_Stock']*reaction_vol_nl)

        # Checks. Numeric hashing matches list equality (1 == 1.0 and hash(1) == hash(1.0)),
        # so the set gives the same accept/reject decisions as v1's `not in combinations`.
        key = tuple(input_data)
        is_new = key not in seen
        fits = (not max_nl) or sum(input_vol) <= max_nl
        if (is_new or not check_repeat) and fits:
            combinations.append(input_data)
            seen.add(key)
            data_point += 1

    # make column name:
    columns_name = []
    for key, value in concentrations_limits.items():
        if not value['Alternatives']:
            columns_name.append(key)
        else:
            columns_name.append(key)
            alternative_name = ['{}_{}'.format(key, i) for i in value['Alternatives']]
            columns_name += alternative_name

    # making csv file
    if make_csv:
        data = pd.DataFrame(np.array(combinations), columns=columns_name)
        data.to_csv('Random_Combination_1.csv', index=False)

    # making dataframe
    if return_df:
        data = pd.DataFrame(np.array(combinations), columns=columns_name)
        return data

    return np.array(combinations)


# transform concentration DataFrame to volume (nanolitre) DataFrame
def concentration_to_volume(concentrations, concentrations_limits, reaction_mixture_vol_nl=10000,
                            fixed_parts={'Lysate': 0.33, 'Saline': 0.1}, round_deg=1, check_water=True):
    """Transform concentrations dataframe to volumes dataframe
       option: add a fixed volumes to all combinations like Lysate
       caution: concentrations unit and metabolite name in concentrations and concentrations_limits must be the same.

    Parameters
    ----------
    concentrations: pandas.DataFrame
        random_combination_generator output

    Returns
    -------
    data: pandas.DataFrame
        a dataframe same as input in shape but volumes data
    """

    # make a copy of original dataframe to avoid further change than can affect that
    data = concentrations.copy(deep=True)
    data_all = data.copy(deep=True)
    data = data[[i for i in data.columns if '_' not in i]]
    data *= reaction_mixture_vol_nl

    for metabolite_name, value in concentrations_limits.items():
        if isinstance(value['Conc_Stock'], Iterable):
            data[metabolite_name] = [round(data[metabolite_name][i] / value['Conc_Stock'][find_stock(value['Conc_Values'], value['Conc_Stock'], data_all[metabolite_name][i])[0]], round_deg) for i in range(len(data[metabolite_name]))]
        else:
            data[metabolite_name] = [round(data[metabolite_name][i] / value['Conc_Stock'], round_deg) for i in range(len(data[metabolite_name]))]

    # add fix parts
    if fixed_parts:
        for key, value in fixed_parts.items():
            data[key] = reaction_mixture_vol_nl * value

    # add water to reach the reaction_mixture_vol_nl
    data['water'] = reaction_mixture_vol_nl - data.sum(axis=1)

    # for low stock concentration that is not possible to make, raise an error
    # stock conc should be set in a way that doesn't raise this error to avoid further debugging
    if check_water and not all(data['water'] >= 0): raise Exception("Oops, too concentrated combination!")

    # add alternative
    # make columns name list:
    columns_name = []
    Type_dic = {}
    Stock_dic = {}
    for key, value in concentrations_limits.items():
        if value['Alternatives']:
            columns_name.append(key)
            columns_name.append('{}_Type'.format(key))
            Type_dic[key] = []
        else:
            columns_name.append(key)
        if isinstance(value['Conc_Stock'], Iterable):
            columns_name.append('{}_Stock_Type'.format(key))
            Stock_dic[key] = []

    # Alternatives
    for key in Type_dic.keys():
        data_type = data_all[[i for i in data_all.columns if '{}_'.format(key) in i]]
        for i in data_type.values:
            Type_dic[key].append(concentrations_limits[key]['Alternatives'][np.where(i == 1.0)[0][0]])

    Type_list = list(Type_dic.keys())
    for key in Type_list:
        Type_dic['{}_Type'.format(key)] = Type_dic.pop(key)

    # Stock
    for key in Stock_dic.keys():
        Stock_dic[key] = [concentrations_limits[key]['Conc_Stock'][find_stock(concentrations_limits[key]['Conc_Values'], concentrations_limits[key]['Conc_Stock'], i)[0]] for i in data_all[key]]

    Stock_list = list(Stock_dic.keys())
    for key in Stock_list:
        Stock_dic['{}_Stock_Type'.format(key)] = Stock_dic.pop(key)

    data_final = pd.concat([data, pd.DataFrame(Type_dic), pd.DataFrame(Stock_dic)], axis=1)
    return data_final[columns_name + list(fixed_parts.keys()) + ['water']]


def check_repetitive(combination, df_main):
    """Check to avoid repetitive combinations

    Parameters
    ----------
    combination:
        combinations that want to be checked

    df_main: pandas.DataFrame
        source dataframe

    Returns
    -------
    boolean:
        True: it exists in df_main
        False: it's not
    """
    comparison_df = df_main.merge(combination, indicator=True, how='outer')
    if 'both' in comparison_df._merge.unique():
        return False
    else:
        return True


def feature_columns(concentrations_limits):
    """Model input columns for a ``concentrations_limits`` dict: each factor, then its alternatives one-hot."""
    columns = []
    for key, value in concentrations_limits.items():
        columns.append(key)
        if value['Alternatives']:
            columns += ['{}_{}'.format(key, i) for i in value['Alternatives']]
    return columns
