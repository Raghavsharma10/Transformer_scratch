def main(args=None):
    """Start application."""
    parser = _parser()

    # Python 2 will error 'too few arguments' if no subcommand is supplied.
    # No such error occurs in Python 3, which makes it feasible to check
    # whether a subcommand was provided (displaying a help message if not).
    # argparse internals vary significantly over the major versions, so it's
    # much easier to just override the args passed to it. In this case, print
    # the usage message if there are no args.
    if args is None and len(sys.argv) <= 1:
        sys.argv.append('--help')

    options = parser.parse_args(args)

    # pass options to subcommand
    options.func(options)

    return 0