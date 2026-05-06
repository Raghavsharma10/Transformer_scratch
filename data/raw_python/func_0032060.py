def get_parser(commands):
    """
    Generate argument parser given a list of subcommand specifications.

    :type commands: list of (str, function, function)
    :arg  commands:
        Each element must be a tuple ``(name, adder, runner)``.

        :param   name: subcommand
        :param  adder: a function takes one object which is an instance
                       of :class:`argparse.ArgumentParser` and add
                       arguments to it
        :param runner: a function takes keyword arguments which must be
                       specified by the arguments parsed by the parser
                       defined by `adder`.  Docstring of this function
                       will be the description of the subcommand.

    """
    parser = argparse.ArgumentParser(
        formatter_class=Formatter,
        description=__doc__,
        epilog=EPILOG,
    )
    subparsers = parser.add_subparsers()

    for (name, adder, runner) in commands:
        subp = subparsers.add_parser(
            name,
            formatter_class=Formatter,
            description=runner.__doc__ and textwrap.dedent(runner.__doc__))
        adder(subp)
        subp.set_defaults(func=runner)

    return parser