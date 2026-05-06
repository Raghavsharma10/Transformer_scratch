def parse_fields(fields, as_dict=False):
    '''
    Given a list of fields (or several other variants of the same),
    return back a consistent, normalized form of the same.

    To forms are currently supported:
        dictionary form: dict 'key' is the field name
                                   and dict 'value' is either 1 (include)
                                   or 0 (exclude).
        list form (other): list values are field names to be included

    If fields passed is one of the following values, it will be assumed
    the user wants to include all fields and thus, we return an empty
    dict or list to indicate this, accordingly:
     * all fields: ['~', None, False, True, {}, []]


    '''
    _fields = {}
    if fields in ['~', None, False, True, {}, []]:
        # all these signify 'all fields'
        _fields = {}
    elif isinstance(fields, dict):
        _fields.update(
            {unicode(k).strip(): int(v) for k, v in fields.iteritems()})
    elif isinstance(fields, basestring):
        _fields.update({unicode(s).strip(): 1 for s in fields.split(',')})
    elif isinstance(fields, (list, tuple)):
        _fields.update({unicode(s).strip(): 1 for s in fields})
    else:
        raise ValueError("invalid fields value")
    if as_dict:
        return _fields
    else:
        return sorted(_fields.keys())