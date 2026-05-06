def main():
    """
    Main function of maya launcher. Parses the arguments and tries to
    launch maya with given them.
    """
    config = build_config()

    parser = argparse.ArgumentParser(
        description="""
        Maya Launcher is a useful script that tries to deal with all
        important system environments maya uses when starting up.

        It aims to streamline the setup process of maya to a simple string
        instead of constantly having to make sure paths are setup correctly.
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        default=None,
        help="""
        file is an optional argument telling maya what file to open with
        the launcher.
        """)

    parser.add_argument(
        '-v', '--version',
        type=str,
        choices=get_executable_choices(dict(config.items(
                                            Config.EXECUTABLES))),
        help="""
        Launch Maya with given version.
        """)

    parser.add_argument(
        '-env', '--environment',
        metavar='env',
        type=str,
        default=config.get(Config.DEFAULTS, 'environment'),
        help="""
        Launch maya with given environemnt, if no environment is specified
        will try to use default value. If not default value is specified
        Maya will behave with factory environment setup.
        """)

    parser.add_argument(
        '-p', '--path',
        metavar='path',
        type=str,
        nargs='+',
        help="""
        Path is an optional argument that takes an unlimited number of paths
        to use for environment creation.

        Useful if you don't want to create a environment variable. Just
        pass the path you want to use.
        """)

    parser.add_argument(
        '-e', '--edit',
        action='store_true',
        help="""
        Edit config file.
        """)

    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help="""
        Start maya in dev mode, autoload on plugins are turned off.
        """)

    # Parse the arguments
    args = parser.parse_args()
    if args.edit:
        return config.edit()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Get executable from either default value in config or given value.
    # If neither exists exit launcher.
    if args.version is None:
        exec_default = config.get(Config.DEFAULTS, 'executable')
        if not exec_default and config.items(Config.EXECUTABLES):
            items = dict(config.items(Config.EXECUTABLES))
            exec_ = items[sorted(items.keys(), reverse=True)[0]]
        else:
            exec_ = exec_default
    else:
        exec_ = config.get(Config.EXECUTABLES, args.version)

    build_maya_environment(config, args.environment, args.path)
    logger.info('\nDone building maya environment, launching: \n{}\n'.format(exec_))
    launch(exec_, args)