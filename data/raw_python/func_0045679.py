def is_generic_list(type_: Type) -> bool:
    """Determines whether a type is a List[...].

    How to do this varies for different Python versions, due to the
    typing library not having a stable API. This functions smooths
    over the differences.

    Args:
        type_: The type to check.

    Returns:
        True iff it's a List[...something...].
    """
    if hasattr(typing, '_GenericAlias'):
        # 3.7
        return (isinstance(type_, typing._GenericAlias) and     # type: ignore
                type_.__origin__ is list)
    else:
        # 3.6 and earlier
        return (isinstance(type_, typing.GenericMeta) and
                type_.__origin__ is List)