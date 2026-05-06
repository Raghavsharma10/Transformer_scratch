def convertallfields(data, commdct, block=None):
    """docstring for convertallfields"""
    # import pdbdb; pdb.set_trace()
    for key in list(data.dt.keys()):
        objs = data.dt[key]
        for i, obj in enumerate(objs):
            key_i = data.dtls.index(key)
            key_comm = commdct[key_i]
            try:
                inblock = block[key_i]
            except TypeError as e:
                inblock = None
            obj = convertfields(key_comm, obj, inblock)
            objs[i] = obj