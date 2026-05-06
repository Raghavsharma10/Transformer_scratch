def build_parser():
    "Build an argparse argument parser to parse the command line."

    parser = argparse.ArgumentParser(
        description="""Coursera OAuth2 client CLI. This tool
        helps users of the Coursera App Platform to programmatically access
        Coursera APIs.""",
        epilog="""Please file bugs on github at:
        https://github.com/coursera/courseraoauth2client/issues. If you
        would like to contribute to this tool's development, check us out at:
        https://github.com/coursera/courseraoauth2client""")
    parser.add_argument('-c', '--config', help='the configuration file to use')
    utils.add_logging_parser(parser)

    # We support multiple subcommands. These subcommands have their own
    # subparsers. Each subcommand should set a default value for the 'func'
    # option. We then call the parsed 'func' function, and execution carries on
    # from there.
    subparsers = parser.add_subparsers()

    commands.config.parser(subparsers)
    commands.version.parser(subparsers)

    return parser