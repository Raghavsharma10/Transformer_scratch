def basic_sobject_to_dict(obj):
    """Converts suds object to dict very quickly.
    Does not serialize date time or normalize key case.
    :param obj: suds object
    :return: dict object
    """
    if not hasattr(obj, '__keylist__'):
        return obj
    data = {}
    fields = obj.__keylist__
    for field in fields:
        val = getattr(obj, field)
        if isinstance(val, list):
            data[field] = []
            for item in val:
                data[field].append(basic_sobject_to_dict(item))
        else:
            data[field] = basic_sobject_to_dict(val)
    return data