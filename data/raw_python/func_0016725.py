def sort(fields):
    '''
    Gets a list of <fields> to sort by.
    Also supports getting a single string for sorting by one field.
    Reverse sort is supported by appending '-' to the field name.
    Example: sort(['age', '-height']) will sort by ascending age and descending height.
    '''
    from pymongo import ASCENDING, DESCENDING
    from bson import SON

    if isinstance(fields, str):
        fields = [fields]
    if not hasattr(fields, '__iter__'):
        raise ValueError("expected a list of strings or a string. not a {}".format(type(fields)))
    
    sort = []
    for field in fields:
        if field.startswith('-'):
            field = field[1:]
            sort.append((field, DESCENDING))
            continue
        elif field.startswith('+'):
            field = field[1:]
        sort.append((field, ASCENDING))
    return {'$sort': SON(sort)}