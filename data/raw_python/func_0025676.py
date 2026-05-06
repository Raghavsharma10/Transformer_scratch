def findExtname(fimg, extname, extver=None):
    """
    Returns the list number of the extension corresponding to EXTNAME given.
    """

    i = 0
    extnum = None
    for chip in fimg:
        hdr = chip.header
        if 'EXTNAME' in hdr:
            if hdr['EXTNAME'].strip() == extname.upper():
                if extver is None or hdr['EXTVER'] == extver:
                    extnum = i
                    break
        i += 1
    return extnum