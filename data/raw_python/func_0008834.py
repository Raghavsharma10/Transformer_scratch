def fix_aips_header(header):
    """
    Search through an image header. If the keywords BMAJ/BMIN/BPA are not set,
    but there are AIPS history cards, then we can populate the BMAJ/BMIN/BPA.
    Fix the header if possible, otherwise don't. Either way, don't complain.


    Parameters
    ----------
    header : HDUHeader
        Fits header which may or may not have AIPS history cards.

    Returns
    -------
    header : HDUHeader
        A header which has BMAJ, BMIN, and BPA keys, as well as a new HISTORY card.
    """
    if 'BMAJ' in header and 'BMIN' in header and 'BPA' in header:
        # The header already has the required keys so there is nothing to do
        return header
    aips_hist = [a for a in header['HISTORY'] if a.startswith("AIPS")]
    if len(aips_hist) == 0:
        # There are no AIPS history items to process
        return header
    for a in aips_hist:
        if "BMAJ" in a:
            # this line looks like
            # 'AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00'
            words = a.split()
            bmaj = float(words[3])
            bmin = float(words[5])
            bpa = float(words[7])
            break
    else:
        # there are AIPS cards but there is no BMAJ/BMIN/BPA
        return header
    header['BMAJ'] = bmaj
    header['BMIN'] = bmin
    header['BPA'] = bpa
    header['HISTORY'] = 'Beam information AIPS->fits by AegeanTools'
    return header