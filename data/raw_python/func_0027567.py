def is_classvar(tp):
    """Test if the type represents a class variable. Examples::

        is_classvar(int) == False
        is_classvar(ClassVar) == True
        is_classvar(ClassVar[int]) == True
        is_classvar(ClassVar[List[T]]) == True
    """
    if NEW_TYPING:
        return (tp is ClassVar or
                isinstance(tp, _GenericAlias) and tp.__origin__ is ClassVar)
    try:
        from typing import _ClassVar
        return type(tp) is _ClassVar
    except:
        # SMA: support for very old typing module <=3.5.3
        return False