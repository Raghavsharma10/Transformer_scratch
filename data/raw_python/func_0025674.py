def removeFile(inlist):
    """
    Utility function for deleting a list of files or a single file.

    This function will automatically delete both files of a GEIS image, just
    like 'iraf.imdelete'.
    """

    if not isinstance(inlist, string_types):
    # We do have a list, so delete all filenames in list.
        # Treat like a list of full filenames
        _ldir = os.listdir('.')
        for f in inlist:
        # Now, check to see if there are wildcards which need to be expanded
            if f.find('*') >= 0 or f.find('?') >= 0:
                # We have a wild card specification
                regpatt = f.replace('?', '.?')
                regpatt = regpatt.replace('*', '.*')
                _reg = re.compile(regpatt)
                for file in _ldir:
                    if _reg.match(file):
                        _remove(file)
            else:
                # This is just a single filename
                _remove(f)
    else:
        # It must be a string then, so treat as a single filename
        _remove(inlist)