def enumeration(values, converter=str, default=''):
    """Return an enumeration string based on the given values.

    The following four examples show the standard output of function
    |enumeration|:

    >>> from hydpy.core.objecttools import enumeration
    >>> enumeration(('text', 3, []))
    'text, 3, and []'
    >>> enumeration(('text', 3))
    'text and 3'
    >>> enumeration(('text',))
    'text'
    >>> enumeration(())
    ''

    All given objects are converted to strings by function |str|, as shown
    by the first two examples.  This behaviour can be changed by another
    function expecting a single argument and returning a string:

    >>> from hydpy.core.objecttools import classname
    >>> enumeration(('text', 3, []), converter=classname)
    'str, int, and list'

    Furthermore, you can define a default string that is returned
    in case an empty iterable is given:

    >>> enumeration((), default='nothing')
    'nothing'
    """
    values = tuple(converter(value) for value in values)
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return ' and '.join(values)
    return ', and '.join((', '.join(values[:-1]), values[-1]))