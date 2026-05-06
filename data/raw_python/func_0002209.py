def reshape_array(data_header, array):
    """Extract the appropriate array shape from the header.

    Can handle taking a data header and either bytes containing data or a StructureData
    instance, which will have binary data as well as some additional information.

    Parameters
    ----------
    array : :class:`numpy.ndarray`
    data_header : Data

    """
    shape = tuple(r.size for r in data_header.section.range)
    if shape:
        return array.reshape(*shape)
    else:
        return array