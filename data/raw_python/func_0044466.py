def get_configuration(basename='scriptabit.cfg', parents=None):
    """Parses and returns the program configuration options,
    taken from a combination of ini-style config file, and
    command line arguments.

    Args:
        basename (str): The base filename.
        parents (list): A list of ArgumentParser objects whose arguments
            should also be included in the configuration parsing. These
            ArgumentParser instances **must** be instantiated with the
            `add_help` argument set to `False`, otherwise the main
            ArgumentParser instance will raise an exception due to duplicate
            help arguments.

    Returns:
        The options object, and a function that can be called to print the help
        text.
    """
    copy_default_config_to_user_directory(basename)

    parser = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter,
        add_help=False,
        parents=parents or [],
        default_config_files=[
            resource_filename(
                Requirement.parse("scriptabit"),
                os.path.join('scriptabit', basename)),
            os.path.join(
                os.path.expanduser("~/.config/scriptabit"),
                basename),
            os.path.join(os.curdir, basename)])

    # logging config file
    parser.add(
        '-lc',
        '--logging-config',
        required=False,
        default='scriptabit_logging.cfg',
        metavar='FILE',
        env_var='SCRIPTABIT_LOGGING_CONFIG',
        help='Logging configuration file')

    # Authentication file section
    parser.add(
        '-as',
        '--auth-section',
        required=False,
        default='habitica',
        help='''Name of the authentication file section containing the Habitica
credentials''')

    parser.add(
        '-url',
        '--habitica-api-url',
        required=False,
        default='https://habitica.com/api/v3/',
        help='''The base Habitica API URL''')

    # plugins
    parser.add(
        '-r',
        '--run',
        required=False,
        help='''Select the plugin to run. Note you can only run a single
plugin at a time. If you specify more than one, then only the
last one will be executed. To chain plugins together, create a
new plugin that combines the effects as required.''')

    parser.add(
        '-ls',
        '--list-plugins',
        required=False,
        action='store_true',
        help='''List available plugins''')

    parser.add(
        '-v',
        '--version',
        required=False,
        action='store_true',
        help='''Display scriptabit version''')

    parser.add(
        '-dr',
        '--dry-run',
        required=False,
        action='store_true',
        help='''Conduct a dry run. No changes are written to online services''')

    parser.add(
        '-n',
        '--max-updates',
        required=False,
        type=int,
        default=0,
        help='''If > 0, this sets a limit on the number of plugin updates.
Note that plugins can still exit before the limit is reached.''')

    parser.add(
        '-uf',
        '--update-frequency',
        required=False,
        type=int,
        default=-1,
        help='''If > 0, this specifies the preferred update frequency in minutes
for plugins that run in the update loop. Note that plugins may ignore or limit
this setting if the value is inappropriate for the specific plugin.''')

    parser.add(
        '-h',
        '--help',
        required=False,
        action='store_true',
        help='''Print help''')

    return parser.parse_known_args()[0], parser.print_help