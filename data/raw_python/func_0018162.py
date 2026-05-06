def run(configobj=None):
    """
    TEAL interface for the `acssum` function.

    """
    acssum(configobj['input'],
           configobj['output'],
           exec_path=configobj['exec_path'],
           time_stamps=configobj['time_stamps'],
           verbose=configobj['verbose'],
           quiet=configobj['quiet']
           )