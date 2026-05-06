def _get_attr_value(instance, attr, default=None):
    """
    Simple helper to get the value of an instance's attribute if it exists.

    If the instance attribute is callable it will be called and the result will
    be returned.

    Optionally accepts a default value to return if the attribute is missing.
    Defaults to `None`

    >>> class Foo(object):
    ...     bar = 'baz'
    ...     def hi(self):
    ...         return 'hi'
    >>> f = Foo()
    >>> _get_attr_value(f, 'bar')
    'baz'
    >>> _get_attr_value(f, 'xyz')

    >>> _get_attr_value(f, 'xyz', False)
    False
    >>> _get_attr_value(f, 'hi')
    'hi'
    """
    value = default
    if hasattr(instance, attr):
        value = getattr(instance, attr)
        if callable(value):
            value = value()
    return value