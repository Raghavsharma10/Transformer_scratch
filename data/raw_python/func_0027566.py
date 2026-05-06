def is_union_type(tp):
    """Test if the type is a union type. Examples::

        is_union_type(int) == False
        is_union_type(Union) == True
        is_union_type(Union[int, int]) == False
        is_union_type(Union[T, int]) == True
    """
    if NEW_TYPING:
        return (tp is Union or
                isinstance(tp, _GenericAlias) and tp.__origin__ is Union)
    try:
        from typing import _Union
        return type(tp) is _Union
    except ImportError:
        # SMA: support for very old typing module <=3.5.3
        return type(tp) is Union or type(tp) is type(Union)