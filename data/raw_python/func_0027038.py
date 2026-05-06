def TypeFactory(type_):
    """
    This function creates a standard form type from a simplified form.

    >>> from datetime import date, datetime
    >>> from pyws.functions.args import TypeFactory
    >>> from pyws.functions.args import String, Integer, Float, Date, DateTime
    >>> TypeFactory(str) == String
    True
    >>> TypeFactory(float) == Float
    True
    >>> TypeFactory(date) == Date
    True
    >>> TypeFactory(datetime) == DateTime
    True

    >>> from operator import attrgetter
    >>> from pyws.functions.args import Dict
    >>> dct = TypeFactory({0: 'HelloWorldDict', 'hello': str, 'world': int})
    >>> issubclass(dct, Dict)
    True
    >>> dct.__name__
    'HelloWorldDict'
    >>> fields = sorted(dct.fields, key=attrgetter('name'))
    >>> len(dct.fields)
    2
    >>> fields[0].name == 'hello'
    True
    >>> fields[0].type == String
    True
    >>> fields[1].name == 'world'
    True
    >>> fields[1].type == Integer
    True

    >>> from pyws.functions.args import List
    >>> lst = TypeFactory([int])
    >>> issubclass(lst, List)
    True
    >>> lst.__name__
    'IntegerList'
    >>> lst.element_type == Integer
    True
    """
    if isinstance(type_, type) and issubclass(type_, Type):
        return type_
    for x in __types__:
        if x.represents(type_):
            return x.get(type_)
    raise UnknownType(type_)