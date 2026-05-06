def generate_scan_configuration_description(scan_parameters):
    '''Generate scan parameter dictionary. This is the only way to dynamically create table with dictionary, cannot be done with tables.IsDescription

    Parameters
    ----------
    scan_parameters : list, tuple
        List of scan parameters names (strings).

    Returns
    -------
    table_description : dict
        Table description.

    Usage
    -----
    pytables.createTable(self.raw_data_file_h5.root, name = 'scan_parameters', description = generate_scan_configuration_description(['PlsrDAC']), title = 'scan_parameters', filters = filter_tables)
    '''
    table_description = np.dtype([(key, tb.StringCol(512, pos=idx)) for idx, key in enumerate(scan_parameters)])
    return table_description