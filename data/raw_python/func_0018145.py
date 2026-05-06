def run(configobj=None):
    """
    TEAL interface for the `acscteforwardmodel` function.

    """
    acscteforwardmodel(configobj['input'],
                       exec_path=configobj['exec_path'],
                       time_stamps=configobj['time_stamps'],
                       verbose=configobj['verbose'],
                       quiet=configobj['quiet'],
                       single_core=configobj['single_core']
                       )