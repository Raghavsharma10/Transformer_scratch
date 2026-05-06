def __add_min_max_value(
        parser,
        basename,
        default_min,
        default_max,
        initial,
        help_template):
    """
    Generates parser entries for options
    with a min, max, and default value.

    Args:
        parser: the parser to use.
        basename: the base option name. Generated options will have flags
            --basename-min, --basename-max, and --basename.
        default_min: the default min value
        default_max: the default max value
        initial: the default initial value
        help_template: the help string template.
            $mmi will be replaced with min, max, or initial.
            $name will be replaced with basename.
    """
    help_template = Template(help_template)

    parser.add(
        '--{0}-min'.format(basename),
        default=default_min,
        type=float,
        required=False,
        help=help_template.substitute(mmi='min', name=basename))

    parser.add(
        '--{0}-max'.format(basename),
        default=default_max,
        type=float,
        required=False,
        help=help_template.substitute(mmi='max', name=basename))

    parser.add(
        '--{0}'.format(basename),
        default=initial,
        type=float,
        required=False,
        help=help_template.substitute(mmi='initial', name=basename))