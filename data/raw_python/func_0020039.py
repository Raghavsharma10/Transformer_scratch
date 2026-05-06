def flat_to_nested(data, instance=None, attname=None,
                   separator=None, loads=None):
    '''Convert a flat representation of a dictionary to
a nested representation. Fields in the flat representation are separated
by the *splitter* parameters.

:parameter data: a flat dictionary of key value pairs.
:parameter instance: optional instance of a model.
:parameter attribute: optional attribute of a model.
:parameter separator: optional separator. Default ``"__"``.
:parameter loads: optional data unserializer.
:rtype: a nested dictionary'''
    separator = separator or JSPLITTER
    val = {}
    flat_vals = {}
    for key, value in iteritems(data):
        if value is None:
            continue
        keys = key.split(separator)
        # first key equal to the attribute name
        if attname:
            if keys.pop(0) != attname:
                continue
        if loads:
            value = loads(value)
        # if an instance is available, inject the flat attribute
        if not keys:
            if value is None:
                val = flat_vals = {}
                break
            else:
                continue
        else:
            flat_vals[key] = value

        d = val
        lk = keys[-1]
        for k in keys[:-1]:
            if k not in d:
                nd = {}
                d[k] = nd
            else:
                nd = d[k]
                if not isinstance(nd, dict):
                    nd = {'': nd}
                    d[k] = nd
            d = nd
        if lk not in d:
            d[lk] = value
        else:
            d[lk][''] = value

    if instance and flat_vals:
        for attr, value in iteritems(flat_vals):
            setattr(instance, attr, value)

    return val