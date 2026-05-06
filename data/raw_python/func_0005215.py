def getattr_in(obj, name):
    """ Finds an in @obj via a period-delimited string @name.
        @obj: (#object)
        @name: (#str) |.|-separated keys to search @obj in
        ..
            obj.deep.attr = 'deep value'
            getattr_in(obj, 'obj.deep.attr')
        ..
        |'deep value'|
    """
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj