def _import_yaml(config_file_path):
    """Return a configuration object
    """
    try:
        logger.info('Importing config %s...', config_file_path)
        with open(config_file_path) as config_file:
            return yaml.safe_load(config_file.read())
    except IOError as ex:
        raise RepexError('{0}: {1} ({2})'.format(
            ERRORS['config_file_not_found'], config_file_path, ex))
    except (yaml.parser.ParserError, yaml.scanner.ScannerError) as ex:
        raise RepexError('{0} ({1})'.format(ERRORS['invalid_yaml'], ex))