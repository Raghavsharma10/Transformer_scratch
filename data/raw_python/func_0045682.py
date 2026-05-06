def generic_type_args(type_: Type) -> List[Type]:
    """Gets the type argument list for the given generic type.

    If you give this function List[int], it will return [int], and
    if you give it Union[int, str] it will give you [int, str]. Note
    that on Python < 3.7, Union[int, bool] collapses to Union[int] and
    then to int; this is already done by the time this function is
    called, so it does not help with that.

    Args:
        type_: The type to get the arguments list of.

    Returns:
        A list of Type objects.
    """
    if hasattr(type_, '__union_params__'):
        # 3.5 Union
        return list(type_.__union_params__)
    return list(type_.__args__)