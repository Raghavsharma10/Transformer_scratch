def construct_listener(outfile=None):
    """Create the listener that prints tweets"""
    if outfile is not None:
        if os.path.exists(outfile):
            raise IOError("File %s already exists" % outfile)
        
        outfile = open(outfile, 'wb')

    return PrintingListener(out=outfile)