def save_raw_data_from_data_queue(data_queue, filename, mode='a', title='', scan_parameters=None):  # mode="r+" to append data, raw_data_file_h5 must exist, "w" to overwrite raw_data_file_h5, "a" to append data, if raw_data_file_h5 does not exist it is created
    '''Writing raw data file from data queue

    If you need to write raw data once in a while this function may make it easy for you.
    '''
    if not scan_parameters:
        scan_parameters = {}
    with open_raw_data_file(filename, mode='a', title='', scan_parameters=list(dict.iterkeys(scan_parameters))) as raw_data_file:
        raw_data_file.append(data_queue, scan_parameters=scan_parameters)