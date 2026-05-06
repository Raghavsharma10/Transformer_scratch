def success(msg, *args, **kwargs):
    '''Display a success message'''
    echo('{0} {1}'.format(green(OK), white(msg)), *args, **kwargs)