def countExtn(fimg, extname='SCI'):
    """
    Return the number of 'extname' extensions, defaulting to counting the
    number of SCI extensions.
    """

    closefits = False
    if isinstance(fimg, string_types):
        fimg = fits.open(fimg)
        closefits = True

    n = 0
    for e in fimg:
        if 'extname' in e.header and e.header['extname'] == extname:
            n += 1

    if closefits:
        fimg.close()

    return n