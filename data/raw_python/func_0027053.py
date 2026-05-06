def ListOf(element_type, element_none_value=None):
    """
    This function creates a list type with element type ``element_type`` and an
    empty element value ``element_none_value``.

    >>> from pyws.functions.args import Integer, ListOf
    >>> lst = ListOf(int)
    >>> issubclass(lst, List)
    True
    >>> lst.__name__
    'IntegerList'
    >>> lst.element_type == Integer
    True
    """
    from pyws.functions.args.types import TypeFactory
    element_type = TypeFactory(element_type)
    return type(element_type.__name__ + 'List', (List,), {
        'element_type': element_type,
        'element_none_value': element_none_value})