def _parser():
    """Parse command-line options."""
    launcher = 'pip%s-utils' % sys.version_info.major

    parser = argparse.ArgumentParser(
        description='%s.' % __description__,
        epilog='See `%s COMMAND --help` for help '
               'on a specific subcommand.' % launcher,
        prog=launcher)
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + __version__)

    subparsers = parser.add_subparsers()

    # dependants
    parser_dependants = subparsers.add_parser(
        'dependants',
        add_help=False,
        help='list dependants of package')
    parser_dependants.add_argument(
        'package',
        metavar='PACKAGE',
        type=_distribution)
    parser_dependants.add_argument(
        '-h', '--help',
        action='help',
        help=argparse.SUPPRESS)
    parser_dependants.set_defaults(
        func=command_dependants)

    # dependents
    parser_dependents = subparsers.add_parser(
        'dependents',
        add_help=False,
        help='list dependents of package')
    parser_dependents.add_argument(
        'package',
        metavar='PACKAGE',
        type=_distribution)
    parser_dependents.add_argument(
        '-i', '--info',
        action='store_true',
        help='show version requirements')
    parser_dependents.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='list dependencies recursively')
    parser_dependents.add_argument(
        '-h', '--help',
        action='help',
        help=argparse.SUPPRESS)
    parser_dependents.set_defaults(
        func=command_dependents)

    # locate
    parser_locate = subparsers.add_parser(
        'locate',
        add_help=False,
        help='identify packages that file belongs to')
    parser_locate.add_argument(
        'file',
        metavar='FILE',
        type=argparse.FileType('r'))
    parser_locate.add_argument(
        '-h', '--help',
        action='help',
        help=argparse.SUPPRESS)
    parser_locate.set_defaults(
        func=command_locate)

    # outdated
    parser_outdated = subparsers.add_parser(
        'outdated',
        add_help=False,
        help='list outdated packages that may be updated')
    parser_outdated.add_argument(
        '-b', '--brief',
        action='store_true',
        help='show package name only')
    group = parser_outdated.add_mutually_exclusive_group()
    group.add_argument(
        '-a', '--all',
        action='store_true',
        help='list all outdated packages')
    group.add_argument(
        '-p', '--pinned',
        action='store_true',
        help='list outdated packages unable to be updated')
    group.add_argument(
        '-U', '--upgrade',
        action='store_true',
        dest='update',
        help='update packages that can be updated'
    )
    parser_outdated.add_argument(
        '-h', '--help',
        action='help',
        help=argparse.SUPPRESS)
    parser_outdated.set_defaults(
        func=command_outdated)

    # parents
    parser_parents = subparsers.add_parser(
        'parents',
        add_help=False,
        help='list packages lacking dependants')
    parser_parents.add_argument(
        '-h', '--help',
        action='help',
        help=argparse.SUPPRESS)
    parser_parents.set_defaults(
        func=command_parents)

    return parser