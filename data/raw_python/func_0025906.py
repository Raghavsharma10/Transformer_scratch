def to_jd(baktun, katun, tun, uinal, kin):
    '''Determine Julian day from Mayan long count'''
    return EPOCH + (baktun * 144000) + (katun * 7200) + (tun * 360) + (uinal * 20) + kin