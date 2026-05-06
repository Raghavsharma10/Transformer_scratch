def popattr(obj, attr, default=NOT_PROVIDED):
    """
    Useful for retrieving an object attr and removing it if it's part of it's
    dict while allowing retrieving from subclass.
    i.e.
    class A:
        a = 'a'
    class B(A):
        b = 'b'
    >>> popattr(B, 'a', None)
    'a'
    >>> A.a
    'a'
    """
    val = getattr(obj, attr, default)
    try:
        delattr(obj, attr)
    except AttributeError:
        if default is NOT_PROVIDED:
            raise
    return val