def warning(msg, *args, **kwargs):
    '''Display a warning message'''
    msg = '{0} {1}'.format(yellow(WARNING), msg)
    echo(msg, *args, **kwargs)