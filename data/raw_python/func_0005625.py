def pretend_option(fn):
    # type: (FunctionType) -> FunctionType
    """ Decorator to add a --pretend option to any click command.

    The value won't be passed down to the command, but rather handled in the
    callback. The value will be accessible through `peltak.core.context` under
    'pretend' if the command needs it. To get the current value you can do:

        >>> from peltak.commands import click, root_cli
        >>> from peltak.core import context
        >>>
        >>> @root_cli.command('my-command')
        >>> @pretend_option
        >>> def my_command():
        ...     pretend = context.get('pretend', False)

    This value will be accessible from anywhere in the code.
    """

    def set_pretend(ctx, param, value):     # pylint: disable=missing-docstring
        # type: (click.Context, str, Any) -> None
        from peltak.core import context
        from peltak.core import shell

        context.set('pretend', value or False)
        if value:
            shell.cprint('<90>{}', _pretend_msg())

    return click.option(
        '--pretend',
        is_flag=True,
        help=("Do not actually do anything, just print shell commands that"
              "would be executed."),
        expose_value=False,
        callback=set_pretend
    )(fn)