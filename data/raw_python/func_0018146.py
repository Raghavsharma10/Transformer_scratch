def run(configobj=None):
    """
    TEAL interface for the `acsccd` function.

    """
    acsccd(configobj['input'],
           exec_path=configobj['exec_path'],
           time_stamps=configobj['time_stamps'],
           verbose=configobj['verbose'],
           quiet=configobj['quiet']  #,
           #dqicorr=configobj['dqicorr'],
           #atodcorr=configobj['atodcorr'],
           #blevcorr=configobj['blevcorr'],
           #biascorr=configobj['biascorr']
           )