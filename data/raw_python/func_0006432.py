def save_configuration_to_hdf5(register, configuration_file, name=''):
    '''Saving configuration to HDF5 file from register object

    Parameters
    ----------
    register : pybar.fei4.register object
    configuration_file : string, file
        Filename of the HDF5 configuration file or file object.
    name : string
        Additional identifier (subgroup). Useful when storing more than one configuration inside a HDF5 file.
    '''
    def save_conf():
        logging.info("Saving configuration: %s" % h5_file.filename)
        register.configuration_file = h5_file.filename
        try:
            configuration_group = h5_file.create_group(h5_file.root, "configuration")
        except tb.NodeError:
            configuration_group = h5_file.root.configuration
        if name:
            try:
                configuration_group = h5_file.create_group(configuration_group, name)
            except tb.NodeError:
                configuration_group = h5_file.root.configuration.name

        # calibration_parameters
        try:
            h5_file.remove_node(configuration_group, name='calibration_parameters')
        except tb.NodeError:
            pass
        calibration_data_table = h5_file.create_table(configuration_group, name='calibration_parameters', description=NameValue, title='calibration_parameters')
        calibration_data_row = calibration_data_table.row
        for key, value in register.calibration_parameters.iteritems():
            calibration_data_row['name'] = key
            calibration_data_row['value'] = str(value)
            calibration_data_row.append()
        calibration_data_table.flush()

        # miscellaneous
        try:
            h5_file.remove_node(configuration_group, name='miscellaneous')
        except tb.NodeError:
            pass
        miscellaneous_data_table = h5_file.create_table(configuration_group, name='miscellaneous', description=NameValue, title='miscellaneous')
        miscellaneous_data_row = miscellaneous_data_table.row
        miscellaneous_data_row['name'] = 'Flavor'
        miscellaneous_data_row['value'] = register.flavor
        miscellaneous_data_row.append()
        miscellaneous_data_row['name'] = 'Chip_ID'
        miscellaneous_data_row['value'] = register.chip_id
        miscellaneous_data_row.append()
        for key, value in register.miscellaneous.iteritems():
            miscellaneous_data_row['name'] = key
            miscellaneous_data_row['value'] = value
            miscellaneous_data_row.append()
        miscellaneous_data_table.flush()

        # global
        try:
            h5_file.remove_node(configuration_group, name='global_register')
        except tb.NodeError:
            pass
        global_data_table = h5_file.create_table(configuration_group, name='global_register', description=NameValue, title='global_register')
        global_data_table_row = global_data_table.row
        global_regs = register.get_global_register_objects(readonly=False)
        for global_reg in sorted(global_regs, key=itemgetter('name')):
            global_data_table_row['name'] = global_reg['name']
            global_data_table_row['value'] = global_reg['value']  # TODO: some function that converts to bin, hex
            global_data_table_row.append()
        global_data_table.flush()

        # pixel
        for pixel_reg in register.pixel_registers.itervalues():
            try:
                h5_file.remove_node(configuration_group, name=pixel_reg['name'])
            except tb.NodeError:
                pass
            data = pixel_reg['value'].T
            atom = tb.Atom.from_dtype(data.dtype)
            ds = h5_file.create_carray(configuration_group, name=pixel_reg['name'], atom=atom, shape=data.shape, title=pixel_reg['name'])
            ds[:] = data

    if isinstance(configuration_file, tb.file.File):
        h5_file = configuration_file
        save_conf()
    else:
        with tb.open_file(configuration_file, mode="a", title='') as h5_file:
            save_conf()