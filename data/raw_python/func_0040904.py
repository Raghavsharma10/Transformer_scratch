def afni_copy(filename):
    ''' creates a ``+orig`` copy of the given dataset and returns the filename as a string '''
    if nl.pkg_available('afni',True):
        afni_filename = "%s+orig" % nl.prefix(filename)
        if not os.path.exists(afni_filename + ".HEAD"):
            nl.calc(filename,'a',prefix=nl.prefix(filename))
        return afni_filename