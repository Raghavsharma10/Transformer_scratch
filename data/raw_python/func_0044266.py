def write_moc_fits_hdu(moc):
    """Create a FITS table HDU representation of a MOC.
    """

    # Ensure data are normalized.
    moc.normalize()

    # Determine whether a 32 or 64 bit column is required.
    if moc.order < 14:
        moc_type = np.int32
        col_type = 'J'
    else:
        moc_type = np.int64
        col_type = 'K'

    # Convert to the NUNIQ value which guarantees that one of the
    # top two bits is set so that the order of the value can be
    # determined.
    nuniq = []
    for (order, cells) in moc:
        uniq_prefix = 4 * (4 ** order)
        for npix in cells:
            nuniq.append(npix + uniq_prefix)

    # Prepare the data, and sort into numerical order.
    nuniq = np.array(nuniq, dtype=moc_type)
    nuniq.sort()

    # Create the FITS file.
    col = fits.Column(name='UNIQ', format=col_type, array=nuniq)

    cols = fits.ColDefs([col])
    rec = fits.FITS_rec.from_columns(cols)
    tbhdu = fits.BinTableHDU(rec)

    # Mandatory Keywords.
    tbhdu.header['PIXTYPE'] = 'HEALPIX'
    tbhdu.header['ORDERING'] = 'NUNIQ'
    tbhdu.header['COORDSYS'] = 'C'
    tbhdu.header['MOCORDER'] = moc.order
    tbhdu.header.comments['PIXTYPE'] = 'HEALPix magic code'
    tbhdu.header.comments['ORDERING'] = 'NUNIQ coding method'
    tbhdu.header.comments['COORDSYS'] = 'ICRS reference frame'
    tbhdu.header.comments['MOCORDER'] = 'MOC resolution (best order)'

    # Optional Keywords.
    tbhdu.header['MOCTOOL'] = 'PyMOC ' + version
    tbhdu.header.comments['MOCTOOL'] = 'Name of MOC generator'
    if moc.type is not None:
        tbhdu.header['MOCTYPE'] = moc.type
        tbhdu.header.comments['MOCTYPE'] = 'Source type (IMAGE or CATALOG)'
    if moc.id is not None:
        tbhdu.header['MOCID'] = moc.id
        tbhdu.header.comments['MOCID'] = 'Identifier of the collection'
    if moc.origin is not None:
        tbhdu.header['ORIGIN'] = moc.origin
        tbhdu.header.comments['ORIGIN'] = 'MOC origin'
    tbhdu.header['DATE'] = datetime.utcnow().replace(
        microsecond=0).isoformat()
    tbhdu.header.comments['DATE'] = 'MOC creation date'
    if moc.name is not None:
        tbhdu.header['EXTNAME'] = moc.name
        tbhdu.header.comments['EXTNAME'] = 'MOC name'

    return tbhdu