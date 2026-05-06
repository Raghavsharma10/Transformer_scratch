def newrawobject(data, commdct, key, block=None, defaultvalues=True):
    """Make a new object for the given key.

    Parameters
    ----------
    data : Eplusdata object
        Data dictionary and list of objects for the entire model.
    commdct : list of dicts
        Comments from the IDD file describing each item type in `data`.
    key : str
        Object type of the object to add (in ALL_CAPS).

    Returns
    -------
    list
        A list of field values for the new object.

    """
    dtls = data.dtls
    key = key.upper()

    key_i = dtls.index(key)
    key_comm = commdct[key_i]
    # set default values
    if defaultvalues:
        obj = [comm.get('default', [''])[0] for comm in key_comm]
    else:
        obj = ['' for comm in key_comm]
    if not block:
        inblock = ['does not start with N'] * len(obj)
    else:
        inblock = block[key_i]
    for i, (f_comm, f_val, f_iddname) in enumerate(zip(key_comm, obj, inblock)):
        if i == 0:
            obj[i] = key
        else:
            obj[i] = convertafield(f_comm, f_val, f_iddname)
    obj = poptrailing(obj)  # remove the blank items in a repeating field.
    return obj