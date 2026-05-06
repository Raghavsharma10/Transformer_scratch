def parse_hardware_aliases_config_file(config_path):
    """
        Parse config file for aliases and automatically register them.

        Returns True if alias file was found and parsed (successfully or unsuccessfully).

        Returns False if alias file was not found.

        Config file is a standard .ini file with a section
    """
    if os.path.exists(config_path):
        logging.info("Parsing alias file {:s}".format(config_path))
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            for section in config.sections():
                device = config.get(section, "device")
                hardware_alias = config.get(section, "hardware_alias")
                display_name = config.get(section, "display_name")
                try:
                    logging.info("Adding alias {:s} for device {:s}, display name: {:s} ".format(hardware_alias, device, display_name))
                    HardwareSourceManager().make_instrument_alias(device, hardware_alias, _(display_name))
                except Exception as e:
                    logging.info("Error creating hardware alias {:s} for device {:s} ".format(hardware_alias, device))
                    logging.info(traceback.format_exc())
        except Exception as e:
            logging.info("Error reading alias file from: " + config_path)
            logging.info(traceback.format_exc())
        return True
    return False