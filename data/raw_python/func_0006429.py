def load_configuration_from_text_file(register, configuration_file):
    '''Loading configuration from text files to register object

    Parameters
    ----------
    register : pybar.fei4.register object
    configuration_file : string
        Full path (directory and filename) of the configuration file. If name is not given, reload configuration from file.
    '''
    logging.info("Loading configuration: %s" % configuration_file)
    register.configuration_file = configuration_file

    config_dict = parse_global_config(register.configuration_file)

    if 'Flavor' in config_dict:
        flavor = config_dict.pop('Flavor').lower()
        if register.flavor:
            pass
        else:
            register.init_fe_type(flavor)
    else:
        if register.flavor:
            pass
        else:
            raise ValueError('Flavor not specified')
    if 'Chip_ID' in config_dict:
        chip_id = config_dict.pop('Chip_ID')
        if register.chip_address:
            pass
        else:
            register.set_chip_address(chip_address=chip_id & 0x7, broadcast=True if chip_id & 0x8 else False)
    elif 'Chip_Address' in config_dict:
        chip_address = config_dict.pop('Chip_Address')
        if register.chip_address:
            pass
        else:
            register.set_chip_address(chip_address)
    else:
        if register.chip_id_initialized:
            pass
        else:
            raise ValueError('Chip address not specified')
    global_registers_configured = []
    pixel_registers_configured = []
    for key in config_dict.keys():
        value = config_dict.pop(key)
        if key in register.global_registers:
            register.set_global_register_value(key, value)
            global_registers_configured.append(key)
        elif key in register.pixel_registers:
            register.set_pixel_register_value(key, value)
            pixel_registers_configured.append(key)
        elif key in register.calibration_parameters:
            register.calibration_parameters[key] = value
        else:
            register.miscellaneous[key] = value

    global_registers = register.get_global_register_attributes('name', readonly=False)
    pixel_registers = register.pixel_registers.keys()
    global_registers_not_configured = set(global_registers).difference(global_registers_configured)
    pixel_registers_not_configured = set(pixel_registers).difference(pixel_registers_configured)
    if global_registers_not_configured:
        logging.warning("Following global register(s) not configured: {}".format(', '.join('\'' + reg + '\'' for reg in global_registers_not_configured)))
    if pixel_registers_not_configured:
        logging.warning("Following pixel register(s) not configured: {}".format(', '.join('\'' + reg + '\'' for reg in pixel_registers_not_configured)))
    if register.miscellaneous:
        logging.warning("Found following unknown parameter(s): {}".format(', '.join('\'' + parameter + '\'' for parameter in register.miscellaneous.iterkeys())))