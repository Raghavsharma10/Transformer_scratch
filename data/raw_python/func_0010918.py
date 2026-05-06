def log_param(name, value):
    '''Log a parameter value to the console.

    Parameters
    ----------
    name : str
        Name of the parameter being logged.
    value : any
        Value of the parameter being logged.
    '''
    log('setting {} = {}', click.style(str(name)),
        click.style(str(value), fg='yellow'))