def parser(subparsers):
    "Build an argparse argument parser to parse the command line."

    # create the parser for the version subcommand.
    parser_version = subparsers.add_parser(
        'version',
        help="Output the version of %(prog)s to the console.")
    parser_version.set_defaults(func=command_version)

    return parser_version