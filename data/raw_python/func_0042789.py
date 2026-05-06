def _assign_to_null(obj, path, value, force=True):
    """
    value IS ASSIGNED TO obj[self.path][key]
    path IS AN ARRAY OF PROPERTY NAMES
    force=False IF YOU PREFER TO use setDefault()
    """
    try:
        if obj is Null:
            return
        if _get(obj, CLASS) is NullType:
            d = _get(obj, "__dict__")
            o = d[OBJ]
            p = d["__key__"]
            s = [p]+path
            return _assign_to_null(o, s, value)

        path0 = path[0]

        if len(path) == 1:
            if force:
                obj[path0] = value
            else:
                _setdefault(obj, path0, value)
            return

        old_value = obj.get(path0)
        if old_value == None:
            if value == None:
                return
            else:
                obj[path0] = old_value = {}

        _assign_to_null(old_value, path[1:], value)
    except Exception as e:
        raise e