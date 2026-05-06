def stsci2(hdulist, filename):
    """For STScI GEIS files, need to do extra steps."""

    # Write output file name to the primary header
    instrument = hdulist[0].header.get('INSTRUME', '')
    if instrument in ("WFPC2", "FOC"):
        hdulist[0].header['FILENAME'] = filename