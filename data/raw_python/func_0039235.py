def error(msg, details=None, *args, **kwargs):
    '''Display an error message with optionnal details'''
    msg = '{0} {1}'.format(red(KO), white(msg))
    if details:
        msg = '\n'.join((msg, safe_unicode(details)))
    echo(format_multiline(msg), *args, **kwargs)