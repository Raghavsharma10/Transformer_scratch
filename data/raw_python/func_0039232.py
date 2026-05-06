def header(msg, *args, **kwargs):
    '''Display an header'''
    msg = ' '.join((yellow(HEADER), white(msg), yellow(HEADER)))
    echo(msg, *args, **kwargs)