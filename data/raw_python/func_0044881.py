def color_exepthook(pdb=0, mode=2):
    """
    Make tracebacks after exceptions colored, verbose, and/or call pdb
    (python cmd line debugger) at the place where the exception occurs
    """

    modus = ['Plain', 'Context', 'Verbose'][mode] # select the mode

    sys.excepthook = ultratb.FormattedTB(mode=modus,
                                    color_scheme='Linux', call_pdb=pdb)