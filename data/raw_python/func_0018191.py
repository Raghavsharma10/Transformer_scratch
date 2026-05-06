def run(configobj=None):
    """
    TEAL interface for the `acsrej` function.

    """
    acsrej(configobj['input'],
           configobj['output'],
           exec_path=configobj['exec_path'],
           time_stamps=configobj['time_stamps'],
           verbose=configobj['verbose'],
           shadcorr=configobj['shadcorr'],
           crrejtab=configobj['crrejtab'],
           crmask=configobj['crmask'],
           scalense=configobj['scalense'],
           initgues=configobj['initgues'],
           skysub=configobj['skysub'],
           crsigmas=configobj['crsigmas'],
           crradius=configobj['crradius'],
           crthresh=configobj['crthresh'],
           badinpdq=configobj['badinpdq'],
           readnoise_only=configobj['readnoise_only'])