def parser_setup():
    """Create ArgumentParser object to parse command line arguments.

    Returns:
        parser (object): containing ArgumentParser data and methods.
    Raises:
        SystemExit: if the user enters invalid args.

    """
    parser = argparse.ArgumentParser(description="Control AWS instances from"
                                     " the command line with: list, start,"
                                     " stop or ssh.", prog='awss',
                                     usage="\tawss {command} [ 'NAME' ] "
                                     "[ '-i ID' ] [ OPTIONS ]\n\t{command} ="
                                     " list | start | stop | ssh")
    parser.add_argument('-v', '--version', action="version",
                        version="awss {0}".format(__version__))

    subparsers = parser.add_subparsers(title="For additional help on"
                                       " command parameters", dest='command',
                                       description="type 'awss {command} -h',"
                                       " where {command} is: list, start,"
                                       " stop or ssh")

    # Parser for LIST command
    parser_list = subparsers.add_parser('list', description="List AWS "
                                        "instances from the command line. "
                                        "'awss list' will list instances"
                                        " specified using combinations of "
                                        "NAME, instance-id and current-state."
                                        "  If no specifications are given, "
                                        " all instances will be listed."
                                        "  ex: 'awss list TEST "
                                        "-r' will list instances named 'TEST'"
                                        " that are currently running.",
                                        usage="\tawss list [none] [NAME] [-i "
                                        "ID] [-r] [-s] [OPTIONS]")
    parser_list.add_argument('instname', nargs='?', metavar='NAME',
                             help='specify instance by name')
    parser_list.add_argument('-i', '--id', action="store",
                             help='specify instance by id')
    parser_list.add_argument('-s', '--stopped', action='store_const',
                             dest="inst_state", const="stopped",
                             help='list stopped instances')
    parser_list.add_argument('-r', '--running', action='store_const',
                             dest="inst_state", const="running",
                             help='list running instances')
    parser_list.add_argument('-d', '--debug', action="count",
                             default=0, help=argparse.SUPPRESS)
    parser_list.set_defaults(func=cmd_list)

    # Parser for START command
    parser_start = subparsers.add_parser('start', usage="\tawss start [NAME]"
                                         " [-i ID] [-h]",
                                         description="Start an AWS instance"
                                         " from the command line.")
    parser_start.add_argument('instname', nargs='?', metavar='NAME',
                              help='specify instance by name')
    parser_start.add_argument('-i', '--id', action="store",
                              help='specify instance-id')
    parser_start.add_argument('-d', '--debug', action="count",
                              default=0, help=argparse.SUPPRESS)
    parser_start.set_defaults(func=cmd_startstop)

    # Parser for STOP command
    parser_stop = subparsers.add_parser('stop', usage="\tawss stop [NAME]"
                                        " [-i ID] [-h]",
                                        description="Stop an AWS instance"
                                        " from the command line.")
    parser_stop.add_argument('instname', nargs='?', metavar='NAME',
                             help='specify instance by name')
    parser_stop.add_argument('-i', '--id', action="store",
                             help='specify instance-id')
    parser_stop.add_argument('-d', '--debug', action="count",
                             default=0, help=argparse.SUPPRESS)
    parser_stop.set_defaults(func=cmd_startstop)

    # Parser for SSH command
    parser_ssh = subparsers.add_parser('ssh', usage="\tawss ssh [NAME]"
                                       " [-i ID] [-u USER] [-p] [-h]",
                                       description="Connect to an AWS i"
                                       "nstance via ssh.")
    parser_ssh.add_argument('instname', nargs='?', metavar='NAME',
                            help='specify instance by name')
    parser_ssh.add_argument('-i', '--id', action="store",
                            help='specify instance-id')
    parser_ssh.add_argument('-u', '--user', action="store",
                            help='override default username for ssh')
    parser_ssh.add_argument('-p', '--nopem', action="store_true",
                            default=False, help='connect without PEM key')
    parser_ssh.add_argument('-d', '--debug', action="count",
                            default=0, help=argparse.SUPPRESS)
    parser_ssh.set_defaults(func=cmd_ssh)
    return parser