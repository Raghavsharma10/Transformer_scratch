def _setup_no_fallback(parser):
    """Add the option, --tox-pyenv-no-fallback.

    If this option is set, do not allow fallback to tox's built-in
    strategy for looking up python executables if the call to `pyenv which`
    by this plugin fails. This will allow the error to raise instead
    of falling back to tox's default behavior.
    """

    cli_dest = 'tox_pyenv_fallback'
    halp = ('If `pyenv which {basepython}` exits non-zero when looking '
            'up the python executable, do not allow fallback to tox\'s '
            'built-in default logic.')
    # Add a command-line option.
    tox_pyenv_group = parser.argparser.add_argument_group(
        title='{0} plugin options'.format(__title__),
    )
    tox_pyenv_group.add_argument(
        '--tox-pyenv-no-fallback', '-F',
        dest=cli_dest,
        default=True,
        action='store_false',
        help=halp
    )

    def _pyenv_fallback(testenv_config, value):
        cli_says = getattr(testenv_config.config.option, cli_dest)
        return cli_says or value

    # Add an equivalent tox.ini [testenv] section option.
    parser.add_testenv_attribute(
        name=cli_dest,
        type="bool",
        postprocess=_pyenv_fallback,
        default=False,
        help=('If `pyenv which {basepython}` exits non-zero when looking '
              'up the python executable, allow fallback to tox\'s '
              'built-in default logic.'),
    )