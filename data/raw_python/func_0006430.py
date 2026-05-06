def load_configuration_from_hdf5(register, configuration_file, node=''):
    '''Loading configuration from HDF5 file to register object

    Parameters
    ----------
    register : pybar.fei4.register object
    configuration_file : string, file
        Filename of the HDF5 configuration file or file object.
    node : string
        Additional identifier (subgroup). Useful when more than one configuration is stored inside a HDF5 file.
    '''
    def load_conf():
        logging.info("Loading configuration: %s" % h5_file.filename)
        register.configuration_file = h5_file.filename
        if node:
            configuration_group = h5_file.root.configuration.node
        else:
            configuration_group = h5_file.root.configuration

        # miscellaneous
        for row in configuration_group.miscellaneous:
            name = row['name']
            try:
                value = literal_eval(row['value'])
            except ValueError:
                value = row['value']
            if name == 'Flavor':
                if register.flavor:
                    pass
                else:
                    register.init_fe_type(value)
            elif name == 'Chip_ID':
                if register.chip_address:
                    pass
                else:
                    register.set_chip_address(chip_address=value & 0x7, broadcast=True if value & 0x8 else False)
            elif name == 'Chip_Address':
                if register.chip_address:
                    pass
                else:
                    register.set_chip_address(chip_address=value, broadcast=False)
            else:
                register.miscellaneous[name] = value

        if register.flavor:
            pass
        else:
            raise ValueError('Flavor not specified')

        if register.chip_id_initialized:
            pass
        else:
            raise ValueError('Chip address not specified')

        # calibration parameters
        for row in configuration_group.calibration_parameters:
            name = row['name']
            value = row['value']
            register.calibration_parameters[name] = literal_eval(value)

        # global
        for row in configuration_group.global_register:
            name = row['name']
            value = row['value']
            register.set_global_register_value(name, literal_eval(value))

        # pixels
        for pixel_reg in h5_file.iter_nodes(configuration_group, 'CArray'):  # ['Enable', 'TDAC', 'C_High', 'C_Low', 'Imon', 'FDAC', 'EnableDigInj']:
            if pixel_reg.name in register.pixel_registers:
                register.set_pixel_register_value(pixel_reg.name, np.asarray(pixel_reg).T)  # np.asarray(h5_file.get_node(configuration_group, name=pixel_reg)).T

    if isinstance(configuration_file, tb.file.File):
        h5_file = configuration_file
        load_conf()
    else:
        with tb.open_file(configuration_file, mode="r", title='') as h5_file:
            load_conf()