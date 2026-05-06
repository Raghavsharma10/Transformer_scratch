def GetSources(ID, darcsec=None, stars_only=False):
    '''
    Grabs the EPIC coordinates from the TPF and searches MAST
    for other EPIC targets within the same aperture.

    :param int ID: The 9-digit :py:obj:`EPIC` number of the target
    :param float darcsec: The search radius in arcseconds. \
           Default is four times the largest dimension of the aperture.
    :param bool stars_only: If :py:obj:`True`, only returns objects \
           explicitly designated as `"stars"` in MAST. Default :py:obj:`False`
    :returns: A list of :py:class:`Source` instances containing \
              other :py:obj:`EPIC` targets within or close to this \
              target's aperture
    '''

    client = kplr.API()
    star = client.k2_star(ID)
    tpf = star.get_target_pixel_files()[0]
    with tpf.open() as f:
        crpix1 = f[2].header['CRPIX1']
        crpix2 = f[2].header['CRPIX2']
        crval1 = f[2].header['CRVAL1']
        crval2 = f[2].header['CRVAL2']
        cdelt1 = f[2].header['CDELT1']
        cdelt2 = f[2].header['CDELT2']
        pc1_1 = f[2].header['PC1_1']
        pc1_2 = f[2].header['PC1_2']
        pc2_1 = f[2].header['PC2_1']
        pc2_2 = f[2].header['PC2_2']
        pc = np.array([[pc1_1, pc1_2], [pc2_1, pc2_2]])
        pc = np.linalg.inv(pc)
        crpix1p = f[2].header['CRPIX1P']
        crpix2p = f[2].header['CRPIX2P']
        crval1p = f[2].header['CRVAL1P']
        crval2p = f[2].header['CRVAL2P']
        cdelt1p = f[2].header['CDELT1P']
        cdelt2p = f[2].header['CDELT2P']
        if darcsec is None:
            darcsec = 4 * max(f[2].data.shape)

    epicid, ra, dec, kepmag = MASTRADec(
        star.k2_ra, star.k2_dec, darcsec, stars_only)
    sources = []
    for i, epic in enumerate(epicid):
        dra = (ra[i] - crval1) * np.cos(np.radians(dec[i])) / cdelt1
        ddec = (dec[i] - crval2) / cdelt2
        sx = pc[0, 0] * dra + pc[0, 1] * ddec + crpix1 + crval1p - 1.0
        sy = pc[1, 0] * dra + pc[1, 1] * ddec + crpix2 + crval2p - 1.0
        sources.append(dict(ID=epic, x=sx, y=sy, mag=kepmag[i],
                            x0=crval1p, y0=crval2p))

    return sources