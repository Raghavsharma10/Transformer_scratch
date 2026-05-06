def warnconfig(action='default'):
    """
    Configure the Python warnings.

    :type action: string
    :param action: The configuration to set. Options are: 'default', 'error', 'ignore', 'always', 'module' and 'once'.
    """

    # If action is 'default'
    if action.lower() == 'default':
        # Change warning settings
        warnings.filterwarnings('default')

    # If action is 'error'
    elif action.lower() == 'error':
        # Change warning settings
        warnings.filterwarnings('error')

    # If action is 'ignore'
    elif action.lower() == 'ignore':
        # Change warning settings
        warnings.filterwarnings('ignore')

    # If action is 'always'
    elif action.lower() == 'always':
        # Change warning settings
        warnings.filterwarnings('always')

    # If action is 'module'
    elif action.lower() == 'module':
        # Change warning settings
        warnings.filterwarnings('module')

    # If action is 'once'
    elif action.lower() == 'once':
        # Change warning settings
        warnings.filterwarnings('once')

    # Raise runtime warning
    raise ValueError("Invalid action specified.")