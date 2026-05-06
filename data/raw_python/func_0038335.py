def scrub(data, units=False):
    """
    For input data [w,f,e] or [w,f] returns the list with NaN, negative, and zero flux
    (and corresponding wavelengths and errors) removed.
    """
    units = [i.unit if hasattr(i, 'unit') else 1 for i in data]
    data = [np.asarray(i.value if hasattr(i, 'unit') else i, dtype=np.float32) for i in data if
            isinstance(i, np.ndarray)]
    data = [i[np.where(~np.isinf(data[1]))] for i in data]
    data = [i[np.where(np.logical_and(data[1] > 0, ~np.isnan(data[1])))] for i in data]
    data = [i[np.unique(data[0], return_index=True)[1]] for i in data]
    return [i[np.lexsort([data[0]])] * Q for i, Q in zip(data, units)] if units else [i[np.lexsort([data[0]])] for i in
                                                                                      data]