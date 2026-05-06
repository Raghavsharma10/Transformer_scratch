def sex2dec(ra, dec):
    '''
    Convert sexadecimal hours to decimal degrees. Adapted from
    `PyKE <http://keplergo.arc.nasa.gov/PyKE.shtml>`_.

    :param float ra: The right ascension
    :param float dec: The declination

    :returns: The same values, but in decimal degrees

    '''

    ra = re.sub('\s+', '|', ra.strip())
    ra = re.sub(':', '|', ra.strip())
    ra = re.sub(';', '|', ra.strip())
    ra = re.sub(',', '|', ra.strip())
    ra = re.sub('-', '|', ra.strip())
    ra = ra.split('|')
    outra = (float(ra[0]) + float(ra[1]) / 60. + float(ra[2]) / 3600.) * 15.0

    dec = re.sub('\s+', '|', dec.strip())
    dec = re.sub(':', '|', dec.strip())
    dec = re.sub(';', '|', dec.strip())
    dec = re.sub(',', '|', dec.strip())
    dec = dec.split('|')

    if float(dec[0]) > 0.0:
        outdec = float(dec[0]) + float(dec[1]) / 60. + float(dec[2]) / 3600.
    else:
        outdec = float(dec[0]) - float(dec[1]) / 60. - float(dec[2]) / 3600.

    return outra, outdec