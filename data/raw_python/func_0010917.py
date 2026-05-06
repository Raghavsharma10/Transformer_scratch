def log(msg, *args, **kwargs):
    '''Log a message to the console.

    Parameters
    ----------
    msg : str
        A string to display on the console. This can contain {}-style
        formatting commands; the remaining positional and keyword arguments
        will be used to fill them in.
    '''
    now = datetime.datetime.now()
    module = 'downhill'
    if _detailed_callsite:
        caller = inspect.stack()[1]
        parts = caller.filename.replace('.py', '').split('/')
        module = '{}:{}'.format(
            '.'.join(parts[parts.index('downhill')+1:]), caller.lineno)
    click.echo(' '.join((
        click.style(now.strftime('%Y%m%d'), fg='blue'),
        click.style(now.strftime('%H%M%S'), fg='cyan'),
        click.style(module, fg='magenta'),
        msg.format(*args, **kwargs),
    )))