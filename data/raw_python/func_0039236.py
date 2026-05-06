def exit_with_error(msg='Aborted', details=None, code=-1, *args, **kwargs):
    '''Exit with error'''
    error(msg, details=details, *args, **kwargs)
    sys.exit(code)