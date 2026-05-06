def makeref2namesdct(name2refdct):
    """make the ref2namesdct in the idd_index"""
    ref2namesdct = {}
    for key, values in name2refdct.items():
        for value in values:
            ref2namesdct.setdefault(value, set()).add(key)
    return ref2namesdct