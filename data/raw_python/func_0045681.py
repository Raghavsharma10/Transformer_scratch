def is_generic_union(type_: Type) -> bool:
    """Determines whether a type is a Union[...].

    How to do this varies for different Python versions, due to the
    typing library not having a stable API. This functions smooths
    over the differences.

    Args:
        type_: The type to check.

    Returns:
        True iff it's a Union[...something...].
    """
    if hasattr(typing, '_GenericAlias'):
        # 3.7
        return (isinstance(type_, typing._GenericAlias) and     # type: ignore
                type_.__origin__ is Union)
    else:
        if hasattr(typing, '_Union'):
            # 3.6
            return isinstance(type_, typing._Union)             # type: ignore
        else:
            # 3.5 and earlier (?)
            return isinstance(type_, typing.UnionMeta)          # type: ignore
    raise RuntimeError('Could not determine whether type is a Union. Is this'
                       ' a YAtiML-supported Python version?')