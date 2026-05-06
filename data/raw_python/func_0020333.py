def GetHiResImage(ID):
    '''
    Queries the Palomar Observatory Sky Survey II catalog to
    obtain a higher resolution optical image of the star with EPIC number
    :py:obj:`ID`.

    '''

    # Get the TPF info
    client = kplr.API()
    star = client.k2_star(ID)
    k2ra = star.k2_ra
    k2dec = star.k2_dec
    tpf = star.get_target_pixel_files()[0]
    with tpf.open() as f:
        k2wcs = WCS(f[2].header)
        shape = np.array(f[1].data.field('FLUX'), dtype='float64')[0].shape

    # Get the POSS URL
    hou = int(k2ra * 24 / 360.)
    min = int(60 * (k2ra * 24 / 360. - hou))
    sec = 60 * (60 * (k2ra * 24 / 360. - hou) - min)
    ra = '%02d+%02d+%.2f' % (hou, min, sec)
    sgn = '' if np.sign(k2dec) >= 0 else '-'
    deg = int(np.abs(k2dec))
    min = int(60 * (np.abs(k2dec) - deg))
    sec = 3600 * (np.abs(k2dec) - deg - min / 60)
    dec = '%s%02d+%02d+%.1f' % (sgn, deg, min, sec)
    url = 'https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&' + \
          'r=%s&d=%s&e=J2000&h=3&w=3&f=fits&c=none&fov=NONE&v3=' % (ra, dec)

    # Query the server
    r = urllib.request.Request(url)
    handler = urllib.request.urlopen(r)
    code = handler.getcode()
    if int(code) != 200:
        # Unavailable
        return None
    data = handler.read()

    # Atomically write to a temp file
    f = NamedTemporaryFile("wb", delete=False)
    f.write(data)
    f.flush()
    os.fsync(f.fileno())
    f.close()

    # Now open the POSS fits file
    with pyfits.open(f.name) as ff:
        img = ff[0].data

    # Map POSS pixels onto K2 pixels
    xy = np.empty((img.shape[0] * img.shape[1], 2))
    z = np.empty(img.shape[0] * img.shape[1])
    pwcs = WCS(f.name)
    k = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ra, dec = pwcs.all_pix2world(float(j), float(i), 0)
            xy[k] = k2wcs.all_world2pix(ra, dec, 0)
            z[k] = img[i, j]
            k += 1

    # Resample
    grid_x, grid_y = np.mgrid[-0.5:shape[1] - 0.5:0.1, -0.5:shape[0] - 0.5:0.1]
    resampled = griddata(xy, z, (grid_x, grid_y), method='cubic')

    # Rotate to align with K2 image. Not sure why, but it is necessary
    resampled = np.rot90(resampled)

    return resampled