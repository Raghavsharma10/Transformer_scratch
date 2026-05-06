def addSubparser(subparsers, subcommand, description):
    """
    Add a subparser with subcommand to the subparsers object
    """
    parser = subparsers.add_parser(
        subcommand, description=description, help=description)
    return parser