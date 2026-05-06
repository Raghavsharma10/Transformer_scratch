def DictOf(name, *fields):
    """
    This function creates a dict type with the specified name and fields.

    >>> from pyws.functions.args import DictOf, Field
    >>> dct = DictOf(
    ...     'HelloWorldDict', Field('hello', str), Field('hello', int))
    >>> issubclass(dct, Dict)
    True
    >>> dct.__name__
    'HelloWorldDict'
    >>> len(dct.fields)
    2
    """
    ret = type(name, (Dict,), {'fields': []})
    #noinspection PyUnresolvedReferences
    ret.add_fields(*fields)
    return ret