def add_options(ctx):
    """
    Add command line options.

    :return: None.
    """
    # Add option
    ctx.add_option(
        '--always',
        action='store_true',
        default=False,
        dest='always',
        help='whether always run tasks.',
    )

    # Add option
    ctx.add_option(
        '--check-import',
        action='store_true',
        default=False,
        dest='check_import',
        help='whether import module for dirty checking.',
    )

    # Add option
    ctx.add_option(
        '--venv',
        dest='venv',
        help=(
            'virtual environment directory relative path relative to top'
            ' directory.'
        ),
    )

    # Add option
    ctx.add_option(
        '--venv-add-version',
        default='1',
        dest='venv_add_version',
        # Convert to int so that the value can be used as boolean
        type=int,
        metavar='0|1',
        help=(
            'whether add full Python version to virtual environment directory'
            ' name. E.g. `.py3.5.1.final.0.64bit`. Default is add.'
        ),
    )

    # Add option
    ctx.add_option(
        '--req',
        default=None,
        dest='req_path',
        help='requirements file relative path relative to top directory.',
    )