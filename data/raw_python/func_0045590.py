def check_key(data_object, key, cardinal=False):
    """
    Update the value of an index key by matching values or getting positionals.
    """
    itype = (int, np.int32, np.int64)
    if not isinstance(key, itype + (slice, tuple, list, np.ndarray)):
        raise KeyError("Unknown key type {} for key {}".format(type(key), key))
    keys = data_object.index.values
    if cardinal and data_object._cardinal is not None:
        keys = data_object[data_object._cardinal[0]].unique()
    elif isinstance(key, itype) and key in keys:
        key = list(sorted(data_object.index.values[key]))
    elif isinstance(key, itype) and key < 0:
        key = list(sorted(data_object.index.values[key]))
    elif isinstance(key, itype):
        key = [key]
    elif isinstance(key, slice):
        key = list(sorted(data_object.index.values[key]))
    elif isinstance(key, (tuple, list, pd.Index)) and not np.all(k in keys for k in key):
        key = list(sorted(data_object.index.values[key]))
    return key