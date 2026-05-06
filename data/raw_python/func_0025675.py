def findKeywordExtn(ft, keyword, value=None):
    """
    This function will return the index of the extension in a multi-extension
    FITS file which contains the desired keyword with the given value.
    """

    i = 0
    extnum = -1
    # Search through all the extensions in the FITS object
    for chip in ft:
        hdr = chip.header
        # Check to make sure the extension has the given keyword
        if keyword in hdr:
            if value is not None:
                # If it does, then does the value match the desired value
                # MUST use 'str.strip' to match against any input string!
                if hdr[keyword].strip() == value:
                    extnum = i
                    break
            else:
                extnum = i
                break
        i += 1
    # Return the index of the extension which contained the
    # desired EXTNAME value.
    return extnum