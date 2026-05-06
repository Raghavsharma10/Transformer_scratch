def create_attr_filter(request, mapped_class):
    """Create an ``and_`` SQLAlchemy filter (a ClauseList object) based
    on the request params (``queryable``, ``eq``, ``ne``, ...).

    Arguments:

    request
        the request.

    mapped_class
        the SQLAlchemy mapped class.
    """

    mapping = {
        'eq': '__eq__',
        'ne': '__ne__',
        'lt': '__lt__',
        'lte': '__le__',
        'gt': '__gt__',
        'gte': '__ge__',
        'like': 'like',
        'ilike': 'ilike'
    }
    filters = []
    if 'queryable' in request.params:
        queryable = request.params['queryable'].split(',')
        for k in request.params:
            if len(request.params[k]) <= 0 or '__' not in k:
                continue
            col, op = k.split("__")
            if col not in queryable or op not in mapping:
                continue
            column = getattr(mapped_class, col)
            f = getattr(column, mapping[op])(request.params[k])
            filters.append(f)
    return and_(*filters) if len(filters) > 0 else None