def from_jd(jd):
    '''Calculate Mayan long count from Julian day'''
    d = jd - EPOCH
    baktun = trunc(d / 144000)
    d = (d % 144000)
    katun = trunc(d / 7200)
    d = (d % 7200)
    tun = trunc(d / 360)
    d = (d % 360)
    uinal = trunc(d / 20)
    kin = int((d % 20))

    return (baktun, katun, tun, uinal, kin)