def stsci(hdulist):
    """For STScI GEIS files, need to do extra steps."""

    instrument = hdulist[0].header.get('INSTRUME', '')

    # Update extension header keywords
    if instrument in ("WFPC2", "FOC"):
        rootname = hdulist[0].header.get('ROOTNAME', '')
        filetype = hdulist[0].header.get('FILETYPE', '')
        for i in range(1, len(hdulist)):
            # Add name and extver attributes to match PyFITS data structure
            hdulist[i].name = filetype
            hdulist[i]._extver = i
            # Add extension keywords for this chip to extension
            hdulist[i].header['EXPNAME'] = (rootname, "9 character exposure identifier")
            hdulist[i].header['EXTVER']= (i, "extension version number")
            hdulist[i].header['EXTNAME'] = (filetype, "extension name")
            hdulist[i].header['INHERIT'] = (True, "inherit the primary header")
            hdulist[i].header['ROOTNAME'] = (rootname, "rootname of the observation set")