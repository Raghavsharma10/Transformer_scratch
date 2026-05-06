def addVars(filename, varNamesStr, outOf=None):
    r"""Like `saveVars`, but appends additional variables to file."""
    filename, varnames, outOf = __saveVarsHelper(filename, varNamesStr, outOf)
    f = None
    try:
        f = open(filename, "rb")
        h = cPickle.load(f)
        f.close()

        h.update(dict(zip(varnames, atIndices(outOf, varnames))))
        f = open(filename, "wb")
        cPickle.dump( h, f , 1 )
    finally:
        if f: f.close()