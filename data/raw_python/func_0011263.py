def secho(text, file=None, nl=True, err=False, color=None, **styles):
    """This function combines :func:`echo` and :func:`style` into one
    call.  As such the following two calls are the same::

        click.secho('Hello World!', fg='green')
        click.echo(click.style('Hello World!', fg='green'))

    All keyword arguments are forwarded to the underlying functions
    depending on which one they go with.

    .. versionadded:: 2.0
    """
    return echo(style(text, **styles), file=file, nl=nl, err=err, color=color)