def update_cached_fields(*args):
    """
    Calls update_cached_fields() for each object passed in as argument.
    Supports also iterable objects by checking __iter__ attribute.
    :param args: List of objects
    :return: None
    """
    for a in args:
        if a is not None:
            if hasattr(a, '__iter__'):
                for e in a:
                    e.update_cached_fields()
            else:
                a.update_cached_fields()