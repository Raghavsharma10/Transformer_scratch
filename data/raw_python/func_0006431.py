def save_configuration_to_text_file(register, configuration_file):
    '''Saving configuration to text files from register object

    Parameters
    ----------
    register : pybar.fei4.register object
    configuration_file : string
        Filename of the configuration file.
    '''
    configuration_path, filename = os.path.split(configuration_file)
    if os.path.split(configuration_path)[1] == 'configs':
        configuration_path = os.path.split(configuration_path)[0]
    filename = os.path.splitext(filename)[0].strip()
    register.configuration_file = os.path.join(os.path.join(configuration_path, 'configs'), filename + ".cfg")
    if os.path.isfile(register.configuration_file):
        logging.warning("Overwriting configuration: %s", register.configuration_file)
    else:
        logging.info("Saving configuration: %s" % register.configuration_file)
    pixel_reg_dict = {}
    for path in ["tdacs", "fdacs", "masks", "configs"]:
        configuration_file_path = os.path.join(configuration_path, path)
        if not os.path.exists(configuration_file_path):
            os.makedirs(configuration_file_path)
        if path == "tdacs":
            dac = register.get_pixel_register_objects(name="TDAC")[0]
            dac_config_path = os.path.join(configuration_file_path, "_".join([dac['name'].lower(), filename]) + ".dat")
            write_pixel_dac_config(dac_config_path, dac['value'])
            pixel_reg_dict[dac['name']] = os.path.relpath(dac_config_path, os.path.dirname(register.configuration_file))
        elif path == "fdacs":
            dac = register.get_pixel_register_objects(name="FDAC")[0]
            dac_config_path = os.path.join(configuration_file_path, "_".join([dac['name'].lower(), filename]) + ".dat")
            write_pixel_dac_config(dac_config_path, dac['value'])
            pixel_reg_dict[dac['name']] = os.path.relpath(dac_config_path, os.path.dirname(register.configuration_file))
        elif path == "masks":
            masks = register.get_pixel_register_objects(bitlength=1)
            for mask in masks:
                dac_config_path = os.path.join(configuration_file_path, "_".join([mask['name'].lower(), filename]) + ".dat")
                write_pixel_mask_config(dac_config_path, mask['value'])
                pixel_reg_dict[mask['name']] = os.path.relpath(dac_config_path, os.path.dirname(register.configuration_file))
        elif path == "configs":
            with open(register.configuration_file, 'w') as f:
                lines = []
                lines.append("# FEI4 Flavor\n")
                lines.append('%s %s\n' % ('Flavor', register.flavor))
                lines.append("\n# FEI4 Chip ID\n")
                lines.append('%s %d\n' % ('Chip_ID', register.chip_id))
                lines.append("\n# FEI4 Global Registers\n")
                global_regs = register.get_global_register_objects(readonly=False)
                for global_reg in sorted(global_regs, key=itemgetter('name')):
                    lines.append('%s %d\n' % (global_reg['name'], global_reg['value']))
                lines.append("\n# FEI4 Pixel Registers\n")
                for key in sorted(pixel_reg_dict):
                    lines.append('%s %s\n' % (key, pixel_reg_dict[key]))
                lines.append("\n# FEI4 Calibration Parameters\n")
                for key in register.calibration_parameters:
                    if register.calibration_parameters[key] is None:
                        lines.append('%s %s\n' % (key, register.calibration_parameters[key]))
                    elif isinstance(register.calibration_parameters[key], (float, int, long)):
                        lines.append('%s %s\n' % (key, round(register.calibration_parameters[key], 4)))
                    elif isinstance(register.calibration_parameters[key], list):
                        lines.append('%s %s\n' % (key, [round(elem, 2) for elem in register.calibration_parameters[key]]))
                    else:
                        raise ValueError('type %s not supported' % type(register.calibration_parameters[key]))
                if register.miscellaneous:
                    lines.append("\n# Miscellaneous\n")
                    for key, value in register.miscellaneous.iteritems():
                        lines.append('%s %s\n' % (key, value))
                f.writelines(lines)