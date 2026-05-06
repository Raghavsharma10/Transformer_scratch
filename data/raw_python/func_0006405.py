def open_raw_data_file(filename, mode="w", title="", scan_parameters=None, socket_address=None):
    '''Mimics pytables.open_file() and stores the configuration and run configuration

    Returns:
    RawDataFile Object

    Examples:
    with open_raw_data_file(filename = self.scan_data_filename, title=self.scan_id, scan_parameters=[scan_parameter]) as raw_data_file:
        # do something here
        raw_data_file.append(self.readout.data, scan_parameters={scan_parameter:scan_parameter_value})
    '''
    return RawDataFile(filename=filename, mode=mode, title=title, scan_parameters=scan_parameters, socket_address=socket_address)