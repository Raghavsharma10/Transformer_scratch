def create_dataset(group, name, data, units='', datatype=DataTypes.UNDEFINED,
                   chunks=True, maxshape=None, compression=None,
                   **attributes):
    """Create an ARF dataset under group, setting required attributes

    Required arguments:
    name --   the name of dataset in which to store the data
    data --   the data to store

    Data can be of the following types:

    * sampled data: an N-D numerical array of measurements
    * "simple" event data: a 1-D array of times
    * "complex" event data: a 1-D array of records, with field 'start' required

    Optional arguments:
    datatype --      a code defining the nature of the data in the channel
    units --         channel units (optional for sampled data, otherwise required)
    sampling_rate -- required for sampled data and event data with units=='samples'

    Arguments passed to h5py:
    maxshape --    make the node resizable up to this shape. Use None for axes that
                   need to be unlimited.
    chunks --      specify the chunk size. The optimal chunk size depends on the
                   intended use of the data. For single-channel sampled data the
                   auto-chunking (True) is probably best.
    compression -- compression strategy. Can be 'gzip', 'szip', 'lzf' or an integer
                   in range(10) specifying gzip(N).  Only gzip is really portable.

    Additional arguments are set as attributes on the created dataset

    Returns the created dataset
    """
    from numpy import asarray
    srate = attributes.get('sampling_rate', None)
    # check data validity before doing anything
    if not hasattr(data, 'dtype'):
        data = asarray(data)
        if data.dtype.kind in ('S', 'O', 'U'):
            raise ValueError(
                "data must be in array with numeric or compound type")
    if data.dtype.kind == 'V':
        if 'start' not in data.dtype.names:
            raise ValueError("complex event data requires 'start' field")
        if not isinstance(units, (list, tuple)):
            raise ValueError("complex event data requires sequence of units")
        if not len(units) == len(data.dtype.names):
            raise ValueError("number of units doesn't match number of fields")
    if units == '':
        if srate is None or not srate > 0:
            raise ValueError(
                "unitless data assumed time series and requires sampling_rate attribute")
    elif units == 'samples':
        if srate is None or not srate > 0:
            raise ValueError(
                "data with units of 'samples' requires sampling_rate attribute")
    # NB: can't really catch case where sampled data has units but doesn't
    # have sampling_rate attribute

    dset = group.create_dataset(
        name, data=data, maxshape=maxshape, chunks=chunks, compression=compression)
    set_attributes(dset, units=units, datatype=datatype, **attributes)
    return dset