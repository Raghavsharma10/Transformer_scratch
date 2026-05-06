def checkSetReadOnly(fname, raiseOnErr = False):
    """ See if we have write-privileges to this file.  If we do, and we
    are not supposed to, then fix that case. """
    if os.access(fname, os.W_OK):
        # We can write to this but it is supposed to be read-only. Fix it.
        # Take away usr-write, leave group and other alone, though it
        # may be simpler to just force/set it to: r--r--r-- or r--------
        irafutils.setWritePrivs(fname, False, ignoreErrors= not raiseOnErr)