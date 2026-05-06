def getitem_in(obj, name):
    """ Finds a key in @obj via a period-delimited string @name.
        @obj: (#dict)
        @name: (#str) |.|-separated keys to search @obj in
        ..
            obj = {'foo': {'bar': {'baz': True}}}
            getitem_in(obj, 'foo.bar.baz')
        ..
        |True|
    """
    for part in name.split('.'):
        obj = obj[part]
    return obj