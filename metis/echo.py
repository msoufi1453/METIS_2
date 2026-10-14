"""Plate layouts and Echo liquid-handler transfer lists (METIS v1 API, unchanged)."""
import pandas as pd


def put_volumes_to_384_wells(volumes_array, starting_well='A1', vertical=False, make_csv=False):
    """Make a dataframe as a 384 well plate for each metabolite
        
    Parameters
    ----------
    volumes_array: 
        a dataframe with columns are component, each row vol of that component (e.g. volumes.csv) 
        
    starting_well: 'A1'
        name of the well in 384 well plates that you want to start filling
    
    vertical:
        if True, it will fill the plate column by column top down
        if False, it will fill the plate row by row, left to right
        
    Returns
    -------
    all_dataframe:
        a list consists of one dataframe for each metabolite that shows appropriate 384 well plate
        
    named_volumes:
        one separate dataframe that adds well name to volume dataframe
    """
    if len(volumes_array) > 384: raise ValueError

    all_dataframe = {}
    rows_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    if not vertical:
        from_well = rows_name.index(starting_well[0]) * 24 + int(starting_well[1:]) - 1
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 25))
            # add data one by one in each row
            # (0, 0)--------->(0,23)
            # .......................
            # (15,0)--------->(15,23)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index // 24, index % 24] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[index // 24], index % 24 + 1) for index in named_volumes.index]
        named_volumes['well_name'] = names

    if vertical:
        from_well = rows_name.index(starting_well[0]) + (int(starting_well[1:]) - 1) * 16
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 25))
            # add data one by one in each column
            # (0, 0)---->-----(0,23)
            # ||||||..........||||||
            # (15,0)---->-----(15,23)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index % 16, index // 16] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[(index + from_well) % 16], (index + from_well) // 16 + 1) for index in
                 named_volumes.index]
        named_volumes['well_name'] = names

    # notice that this function output two value
    return named_volumes, all_dataframe

def put_volumes_to_96_wells(volumes_array, starting_well='A1', vertical=False, make_csv=False):
    """Make a dataframe as a 96 well plate for each metabolite
        
    Parameters
    ----------
    volumes_array: 
        a dataframe with columns are component, each row vol of that component (e.g. volumes.csv) 
        
    starting_well: 'A1'
        name of the well in 96 well plates that you want to start filling
    
    vertical:
        if True, it will fill the plate column by column top down
        if False, it will fill the plate row by row, left to right
        
    Returns
    -------
    all_dataframe:
        a list consists of one dataframe for each metabolite that shows appropriate 384 well plate
        
    named_volumes:
        one separate dataframe that adds well name to volume dataframe
    """
    if len(volumes_array) > 96: raise ValueError

    all_dataframe = {}
    rows_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    if not vertical:
        from_well = rows_name.index(starting_well[0]) * 12 + int(starting_well[1:]) - 1
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 13))
            # add data one by one in each row
            # (0, 0)--------->(0,11)
            # .......................
            # (7,0)--------->(7,11)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index // 12, index % 12] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[index // 12], index % 12 + 1) for index in named_volumes.index]
        named_volumes['well_name'] = names

    if vertical:
        from_well = rows_name.index(starting_well[0]) + (int(starting_well[1:]) - 1) * 8
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 13))
            # add data one by one in each column
            # (0, 0)---->-----(0,11)
            # ||||||..........||||||
            # (7,0)---->-----(7,11)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index % 8, index // 8] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[(index + from_well) % 8], (index + from_well) // 8 + 1) for index in
                 named_volumes.index]
        named_volumes['well_name'] = names

    # notice that this function output two value
    return named_volumes, all_dataframe


# make source to destination dataframe for ECHO machine
def source_to_destination(named_volumes, desired_order=None, reset_index=True, check_zero=False):
    """Make a dataframe as a 384/96 well plate for each metabolite
        
    Parameters
    ----------
    named_volume: 
         first output of put_volumes_to_384_wells or put_volumes_to_96_wells function
        
    Returns
    -------
    all_sources:
        separate dataframe for each metabolite that appended in a dict
    
    aggregated:
        aggregated all_sources to one CSV file by your desired order
    """
    all_sources = {}
    for metabolite_name in named_volumes.drop(columns=['well_name']):
        transfers = {'Source_Plate_Barcode': [], 'Source_Well': [], 'Destination_Plate_Barcode': [],
                     'Destination_Well': [], 'Transfer_Volume': []}
        for index in range(len(named_volumes)):
            if named_volumes.loc[index, metabolite_name] > 0 or check_zero == False:
                transfers['Source_Plate_Barcode'].append('Plate1')
                transfers['Source_Well'].append('{} well'.format(metabolite_name))
                transfers['Destination_Plate_Barcode'].append('destPlate1')
                transfers['Destination_Well'].append(named_volumes.loc[index, 'well_name'])
                transfers['Transfer_Volume'].append(named_volumes.loc[index, metabolite_name])
        transfers = pd.DataFrame(transfers)

        all_sources[metabolite_name] = transfers

    # aggregate all dataframe
    aggregated = pd.concat(all_sources.values())

    if desired_order:
        aggregated = pd.concat([all_sources[i] for i in desired_order])

    if reset_index:
        aggregated = aggregated.reset_index(drop=True)

    return all_sources, aggregated


def put_volumes_to_wells(volumes_array, plate_384_well=True, vertical=True, triplicate=False, starting_well='A1', make_csv=False):
    """it's a helper function for put_volumes_to_96_wells and put_volumes_to_384_wells that take care of creaating triplicate
        
    Parameters
    ----------
    volumes_array: 
        a dataframe with columns are component, each row vol of that component (e.g. volumes.csv) 
        
    starting_well: 'A1'
        name of the well in 96 well plates that you want to start filling
    
    vertical:
        if True, it will fill the plate column by column top down
        if False, it will fill the plate row by row, left to right
        
    Returns
    -------

    named_volumes:
        one separate dataframe that adds well name to volume dataframe
    """
    if plate_384_well:
        if triplicate == False:
            intermediate, _ = put_volumes_to_384_wells(volumes_array, starting_well=starting_well, vertical=vertical, make_csv=make_csv)
        else:
            intermediate_1, _ = put_volumes_to_384_wells(volumes_array, starting_well='A1', vertical=vertical, make_csv=make_csv)
            intermediate_2, _ = put_volumes_to_384_wells(volumes_array, starting_well='A9', vertical=vertical, make_csv=make_csv)
            intermediate_3, _ = put_volumes_to_384_wells(volumes_array, starting_well='A17', vertical=vertical, make_csv=make_csv)
            intermediate = pd.concat([intermediate_1, intermediate_2, intermediate_3]).reset_index(drop=True)
    else:
        if triplicate == False:
            intermediate, _ = put_volumes_to_96_wells(volumes_array, starting_well=starting_well, vertical=vertical, make_csv=make_csv)
        else:
            intermediate_1, _ = put_volumes_to_96_wells(volumes_array, starting_well='A1', vertical=vertical, make_csv=make_csv)
            intermediate_2, _ = put_volumes_to_96_wells(volumes_array, starting_well='A5', vertical=vertical, make_csv=make_csv)
            intermediate_3, _ = put_volumes_to_96_wells(volumes_array, starting_well='A9', vertical=vertical, make_csv=make_csv)
            intermediate = pd.concat([intermediate_1, intermediate_2, intermediate_3]).reset_index(drop=True)
    
    return intermediate
