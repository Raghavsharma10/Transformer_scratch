def unite_dict(a, b):
    """
    >>> a = {'name': 'Sylvanas'}
    >>> b = {'gender': 'Man'}
    >>> unite_dict(a, b)
    {'name': 'Sylvanas', 'gender': 'Man'}
    """
    c = {}
    c.update(a)
    c.update(b)
    return c