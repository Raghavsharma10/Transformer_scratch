def check(qpi_or_h5file, checks=["attributes", "background"]):
    """Checks various properties of a :class:`qpimage.core.QPImage` instance

    Parameters
    ----------
    qpi_or_h5file: qpimage.core.QPImage or str
        A QPImage object or a path to an hdf5 file
    checks: list of str
        Which checks to perform ("attributes" and/or "background")

    Raises
    ------
    IntegrityCheckError
        if the checks fail
    """
    if isinstance(checks, str):
        checks = [checks]
    for ch in checks:
        if ch not in ["attributes", "background"]:
            raise ValueError("Unknown check: {}".format(check))

    if isinstance(qpi_or_h5file, QPImage):
        qpi = qpi_or_h5file
    else:
        qpi = QPImage(h5file=qpi_or_h5file, h5mode="r")

    # check attributes
    if "attributes" in checks:
        check_attributes(qpi)

    # check background estimation
    if "background" in checks:
        check_background(qpi)