def save_configuration_dict(h5_file, configuation_name, configuration, **kwargs):
    '''Stores any configuration dictionary to HDF5 file.

    Parameters
    ----------
    h5_file : string, file
        Filename of the HDF5 configuration file or file object.
    configuation_name : str
        Configuration name. Will be used for table name.
    configuration : dict
        Configuration dictionary.
    '''
    def save_conf():
        try:
            h5_file.remove_node(h5_file.root.configuration, name=configuation_name)
        except tb.NodeError:
            pass
        try:
            configuration_group = h5_file.create_group(h5_file.root, "configuration")
        except tb.NodeError:
            configuration_group = h5_file.root.configuration

        scan_param_table = h5_file.create_table(configuration_group, name=configuation_name, description=NameValue, title=configuation_name)
        row_scan_param = scan_param_table.row
        for key, value in dict.iteritems(configuration):
            row_scan_param['name'] = key
            row_scan_param['value'] = str(value)
            row_scan_param.append()
        scan_param_table.flush()

    if isinstance(h5_file, tb.file.File):
        save_conf()
    else:
        if os.path.splitext(h5_file)[1].strip().lower() != ".h5":
            h5_file = os.path.splitext(h5_file)[0] + ".h5"
        with tb.open_file(h5_file, mode="a", title='', **kwargs) as h5_file:
            save_conf()