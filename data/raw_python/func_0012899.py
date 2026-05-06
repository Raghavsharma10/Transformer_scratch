def convertfields_old(key_comm, obj, inblock=None):
    """convert the float and interger fields"""
    convinidd = ConvInIDD()
    typefunc = dict(integer=convinidd.integer, real=convinidd.real)
    types = []
    for comm in key_comm:
        types.append(comm.get('type', [None])[0])
    convs = [typefunc.get(typ, convinidd.no_type) for typ in types]
    try:
        inblock = list(inblock)
    except TypeError as e:
        inblock = ['does not start with N'] * len(obj)
    for i, (val, conv, avar) in enumerate(zip(obj, convs, inblock)):
        if i == 0:
            # inblock[0] is the key
            pass
        else:
            val = conv(val, inblock[i])
        obj[i] = val
    return obj