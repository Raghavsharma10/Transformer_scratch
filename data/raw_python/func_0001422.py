def check_attributes(qpi):
    """Check QPimage attributes

    Parameters
    ----------
    qpi: qpimage.core.QPImage

    Raises
    ------
    IntegrityCheckError
        if the check fails
    """
    missing_attrs = []
    for key in DATA_KEYS:
        if key not in qpi.meta:
            missing_attrs.append(key)
    if missing_attrs:
        msg = "Attributes are missing: {} ".format(missing_attrs) \
            + "in {}!".format(qpi)
        raise IntegrityCheckError(msg)