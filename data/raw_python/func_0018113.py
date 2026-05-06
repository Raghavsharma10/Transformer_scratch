def extract_ref(scihdu, refhdu):
    """Extract section of the reference image that
    corresponds to the given science image.

    This only returns a view, not a copy of the
    reference image's array.

    Parameters
    ----------
    scihdu, refhdu : obj
        Extension HDU's of the science and reference image,
        respectively.

    Returns
    -------
    refdata : array-like
        Section of the relevant reference image.

    Raises
    ------
    NotImplementedError
        Either science or reference data are binned.

    ValueError
        Extracted section size mismatch.

    """
    same_size, rx, ry, x0, y0 = find_line(scihdu, refhdu)

    # Use the whole reference image
    if same_size:
        return refhdu.data

    # Binned data
    if rx != 1 or ry != 1:
        raise NotImplementedError(
            'Either science or reference data are binned')

    # Extract a view of the sub-section
    ny, nx = scihdu.data.shape
    refdata = refhdu.data[y0:y0+ny, x0:x0+nx]

    if refdata.shape != (ny, nx):
        raise ValueError('Extracted reference image is {0} but science image '
                         'is {1}'.format(refdata.shape, (ny, nx)))

    return refdata