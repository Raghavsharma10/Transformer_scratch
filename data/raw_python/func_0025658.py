def verifyWriteMode(files):
    """
    Checks whether files are writable. It is up to the calling routine to raise
    an Exception, if desired.

    This function returns True, if all files are writable and False, if any are
    not writable.  In addition, for all files found to not be writable, it will
    print out the list of names of affected files.
    """

    # Start by insuring that input is a list of filenames,
    # if only a single filename has been given as input,
    # convert it to a list with len == 1.
    if not isinstance(files, list):
        files = [files]

    # Keep track of the name of each file which is not writable
    not_writable = []
    writable = True

    # Check each file in input list
    for fname in files:
        try:
            f = open(fname,'a')
            f.close()
            del f
        except:
            not_writable.append(fname)
            writable = False

    if not writable:
        print('The following file(s) do not have write permission!')
        for fname in not_writable:
            print('    ', fname)

    return writable