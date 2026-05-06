def MASTRADec(ra, dec, darcsec, stars_only=False):
    '''
    Detector location retrieval based upon RA and Dec.
    Adapted from `PyKE <http://keplergo.arc.nasa.gov/PyKE.shtml>`_.

    '''

    # coordinate limits
    darcsec /= 3600.0
    ra1 = ra - darcsec / np.cos(dec * np.pi / 180)
    ra2 = ra + darcsec / np.cos(dec * np.pi / 180)
    dec1 = dec - darcsec
    dec2 = dec + darcsec

    # build mast query
    url = 'http://archive.stsci.edu/k2/epic/search.php?'
    url += 'action=Search'
    url += '&k2_ra=' + str(ra1) + '..' + str(ra2)
    url += '&k2_dec=' + str(dec1) + '..' + str(dec2)
    url += '&max_records=10000'
    url += '&selectedColumnsCsv=id,k2_ra,k2_dec,kp'
    url += '&outputformat=CSV'
    if stars_only:
        url += '&ktc_target_type=LC'
        url += '&objtype=star'

    # retrieve results from MAST
    try:
        lines = urllib.request.urlopen(url)
    except:
        log.warn('Unable to retrieve source data from MAST.')
        lines = ''

    # collate nearby sources
    epicid = []
    kepmag = []
    ra = []
    dec = []
    for line in lines:

        line = line.strip().decode('ascii')

        if (len(line) > 0 and 'EPIC' not in line and 'integer' not in line and
                              'no rows found' not in line):

            out = line.split(',')
            r, d = sex2dec(out[1], out[2])
            epicid.append(int(out[0]))
            kepmag.append(float(out[3]))
            ra.append(r)
            dec.append(d)

    epicid = np.array(epicid)
    kepmag = np.array(kepmag)
    ra = np.array(ra)
    dec = np.array(dec)

    return epicid, ra, dec, kepmag