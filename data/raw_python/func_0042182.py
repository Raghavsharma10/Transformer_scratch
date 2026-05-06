def _getdefault(obj, key):
    """
    obj MUST BE A DICT
    key IS EXPECTED TO BE LITERAL (NO ESCAPING)
    TRY BOTH ATTRIBUTE AND ITEM ACCESS, OR RETURN Null
    """
    try:
        return obj[key]
    except Exception as f:
        pass

    try:
        return getattr(obj, key)
    except Exception as f:
        pass


    try:
        if float(key) == round(float(key), 0):
            return obj[int(key)]
    except Exception as f:
        pass


    # TODO: FIGURE OUT WHY THIS WAS EVER HERE (AND MAKE A TEST)
    # try:
    #     return eval("obj."+text_type(key))
    # except Exception as f:
    #     pass
    return NullType(obj, key)