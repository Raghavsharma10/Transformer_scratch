def echo(msg, *args, **kwargs):
    '''Wraps click.echo, handles formatting and check encoding'''
    file = kwargs.pop('file', None)
    nl = kwargs.pop('nl', True)
    err = kwargs.pop('err', False)
    color = kwargs.pop('color', None)
    msg = safe_unicode(msg).format(*args, **kwargs)
    click.echo(msg, file=file, nl=nl, err=err, color=color)