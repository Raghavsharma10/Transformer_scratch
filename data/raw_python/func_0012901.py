def convertfields(key_comm, obj, inblock=None):
    """convert based on float, integer, and A1, N1"""
    # f_ stands for field_
    convinidd = ConvInIDD()
    if not inblock:
        inblock = ['does not start with N'] * len(obj)
    for i, (f_comm, f_val, f_iddname) in enumerate(zip(key_comm, obj, inblock)):
        if i == 0:
            # inblock[0] is the iddobject key. No conversion here
            pass
        else:
            obj[i] = convertafield(f_comm, f_val, f_iddname)
    return obj