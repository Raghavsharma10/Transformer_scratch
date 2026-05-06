def run(configobj=None):
    """
    TEAL interface for the `calacs` function.

    """
    calacs(configobj['input_file'],
           exec_path=configobj['exec_path'],
           time_stamps=configobj['time_stamps'],
           temp_files=configobj['temp_files'],
           verbose=configobj['verbose'],
           debug=configobj['debug'],
           quiet=configobj['quiet'],
           single_core=configobj['single_core']
           )