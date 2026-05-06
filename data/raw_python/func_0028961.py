def configure_app(**kwargs):
    """Builds up the settings using the same method as logan"""
    sys_args = sys.argv
    args, command, command_args = parse_args(sys_args[1:])
    parser = OptionParser()
    parser.add_option('--config', metavar='CONFIG')
    (options, logan_args) = parser.parse_args(args)
    config_path = options.config
    logan_configure(config_path=config_path, **kwargs)