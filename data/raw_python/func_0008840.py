def search_beam(hdulist):
    """
    Will search the beam info from the HISTORY
    :param hdulist:
    :return:
    """
    header = hdulist[0].header
    history = header['HISTORY']
    history_str = str(history)
    #AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00
    if 'BMAJ' in history_str:
        return True
    else:
        return False