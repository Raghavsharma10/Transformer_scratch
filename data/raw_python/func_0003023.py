def parse_config(args):
    """
    Try to load config, to load other journal locations
    Otherwise, return default location

    Returns journal location
    """
    # Try user config or return default location early
    config_path = path.expanduser(args.config_file)
    if not path.exists(config_path):
        # Complain if they provided non-existant config
        if args.config_file != DEFAULT_JOURNAL_RC:
            print("journal: error: config file '" + args.config_file + "' not found")
            sys.exit()
        else:
            # If no config file, use default journal location
            return DEFAULT_JOURNAL

    # If we get here, assume valid config file
    config = ConfigParser.SafeConfigParser({
        'journal':{'default':'__journal'},
        '__journal':{'location':DEFAULT_JOURNAL}
    })
    config.read(config_path)

    journal_location = config.get(config.get('journal', 'default'), 'location');
    if args.journal:
        journal_location = config.get(args.journal, 'location');
    return journal_location