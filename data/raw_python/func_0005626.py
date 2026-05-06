def verbose_option(fn):
    """ Decorator to add a --verbose option to any click command.

    The value won't be passed down to the command, but rather handled in the
    callback. The value will be accessible through `peltak.core.context` under
    'verbose' if the command needs it. To get the current value you can do:

        >>> from peltak.core import context
        >>>
        >>> pretend = context.get('verbose', False)

    This value will be accessible from anywhere in the code.
    """

    def set_verbose(ctx, param, value):     # pylint: disable=missing-docstring
        # type: (click.Context, str, Any) -> None
        from peltak.core import context
        context.set('verbose', value or 0)

    return click.option(
        '-v', '--verbose',
        is_flag=True,
        expose_value=False,
        callback=set_verbose,
        help="Be verbose. Can specify multiple times for more verbosity.",
    )(fn)