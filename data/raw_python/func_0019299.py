def extract(values, types, skip=False):
    """Return a generator that extracts certain objects from `values`.

    This function is thought for supporting the definition of functions
    with arguments, that can be objects of of contain types or that can
    be iterables containing these objects.

    The following examples show that function |extract|
    basically implements a type specific flattening mechanism:

    >>> from hydpy.core.objecttools import extract
    >>> tuple(extract('str1', (str, int)))
    ('str1',)
    >>> tuple(extract(['str1', 'str2'], (str, int)))
    ('str1', 'str2')
    >>> tuple(extract((['str1', 'str2'], [1,]), (str, int)))
    ('str1', 'str2', 1)

    If an object is neither iterable nor of the required type, the
    following exception is raised:

    >>> tuple(extract((['str1', 'str2'], [None, 1]), (str, int)))
    Traceback (most recent call last):
    ...
    TypeError: The given value `None` is neither iterable nor \
an instance of the following classes: str and int.

    Optionally, |None| values can be skipped:

    >>> tuple(extract(None, (str, int), True))
    ()
    >>> tuple(extract((['str1', 'str2'], [None, 1]), (str, int), True))
    ('str1', 'str2', 1)
    """
    if isinstance(values, types):
        yield values
    elif skip and (values is None):
        return
    else:
        try:
            for value in values:
                for subvalue in extract(value, types, skip):
                    yield subvalue
        except TypeError as exc:
            if exc.args[0].startswith('The given value'):
                raise exc
            else:
                raise TypeError(
                    f'The given value `{repr(values)}` is neither iterable '
                    f'nor an instance of the following classes: '
                    f'{enumeration(types, converter=instancename)}.')