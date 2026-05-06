def _read_config(filename):
    """Read configuration from the given file.

    Parsing is performed through the configparser library.

    Returns:
        dict: a flattened dict of (option_name, value), using defaults.
    """
    parser = configparser.RawConfigParser()
    if filename and not parser.read(filename):
        sys.stderr.write("Unable to open configuration file %s. Use --config='' to disable this warning.\n" % filename)

    config = {}

    for section, defaults in BASE_CONFIG.items():
        # Patterns are handled separately
        if section == 'patterns':
            continue

        for name, descr in defaults.items():
            kind, default = descr
            if section in parser.sections() and name in parser.options(section):
                if kind == 'int':
                    value = parser.getint(section, name)
                elif kind == 'float':
                    value = parser.getfloat(section, name)
                elif kind == 'bool':
                    value = parser.getboolean(section, name)
                else:
                    value = parser.get(section, name)
            else:
                value = default
            config[name] = value

    if 'patterns' in parser.sections():
        patterns = [parser.get('patterns', opt) for opt in parser.options('patterns')]
    else:
        patterns = DEFAULT_PATTERNS
    config['patterns'] = patterns

    return config