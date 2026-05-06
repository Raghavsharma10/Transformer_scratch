def ip_extra_syshook(fnc, pdb=0, filename=None):
    """
    Extended system hook for exceptions.

    supports logging of tracebacks to a file

    lets fnc() be executed imediately before the IPython
    Verbose Traceback is started

    this can be used to pop up a QTMessageBox: "An exception occured"
    """

    assert isinstance(fnc, collections.Callable)
    from IPython.core import ultratb
    import time

    if not filename == None:
        assert isinstance(filename, str)
        pdb = 0

    ip_excepthook = ultratb.FormattedTB(mode='Verbose',
                                    color_scheme='Linux', call_pdb=pdb)

    fileTraceback = ultratb.FormattedTB(mode='Verbose',
                                    color_scheme='NoColor', call_pdb=0)

    # define the new excepthook
    def theexecpthook (type, value, traceback):
        fnc()
        ip_excepthook(type, value, traceback)
        # write this to a File without Colors
        if not filename == None:
            outFile = open(filename, "a")
            outFile.write("--" + time.ctime()+" --\n")
            outFile.write(fileTraceback.text(type, value, traceback))
            outFile.write("\n-- --\n")
            outFile.close()

    # assign it
    sys.excepthook = theexecpthook