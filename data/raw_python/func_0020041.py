def addmul_number_dicts(series):
    '''Multiply dictionaries by a numeric values and add them together.

:parameter series: a tuple of two elements tuples. Each serie is of the form::

        (weight,dictionary)

    where ``weight`` is a number and ``dictionary`` is a dictionary with
    numeric values.
:parameter skip: optional list of field names to skip.

Only common fields are aggregated. If a field has a non-numeric value it is
not included either.'''
    if not series:
        return
    vtype = value_type((s[1] for s in series))
    if vtype == 1:
        return sum((weight*float(d) for weight, d in series))
    elif vtype == 3:
        keys = set(series[0][1])
        for serie in series[1:]:
            keys.intersection_update(serie[1])
        results = {}
        for key in keys:
            key_series = tuple((weight, d[key]) for weight, d in series)
            result = addmul_number_dicts(key_series)
            if result is not None:
                results[key] = result
        return results