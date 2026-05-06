def append_data(dset, data):
    """Append data to dset along axis 0. Data must be a single element or
    a 1D array of the same type as the dataset (including compound datatypes)."""
    N = data.shape[0] if hasattr(data, 'shape') else 1
    if N == 0:
        return
    oldlen = dset.shape[0]
    newlen = oldlen + N
    dset.resize(newlen, axis=0)
    dset[oldlen:] = data