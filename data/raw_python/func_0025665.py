def buildFITSName(geisname):
    """Build a new FITS filename for a GEIS input image."""

    # User wants to make a FITS copy and update it...
    _indx = geisname.rfind('.')
    _fitsname = geisname[:_indx] + '_' + geisname[_indx + 1:-1] + 'h.fits'

    return _fitsname