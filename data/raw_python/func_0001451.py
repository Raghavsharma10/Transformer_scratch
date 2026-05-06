def main(cmd_args: list = None) -> None:
    """
    :cmd_args: An optional list of command line arguments.

    Main function of chronos CLI tool.
    """
    parser = argparse.ArgumentParser(description='Auto-versioning utility.')
    subparsers = parser.add_subparsers()

    infer_parser = subparsers.add_parser('infer', help='Infers next version.')
    infer_parser.set_defaults(func=infer)

    commit_parser = subparsers.add_parser('commit',
                                          help='Makes release commit.')
    commit_parser.set_defaults(func=commit)

    bump_parser = subparsers.add_parser('bump', help='Bumps version.')
    bump_parser.add_argument('type', nargs='?', default='patch',
                             choices=['patch', 'minor', 'major'],
                             help='The type of version to bump.')
    bump_parser.set_defaults(func=bump)

    try:
        if cmd_args:
            args = parser.parse_args(cmd_args)
        else:
            args = parser.parse_args()

        args.func(args)
    except AttributeError:
        parser.print_help()