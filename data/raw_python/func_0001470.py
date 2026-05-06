def _collapse_preconditions(base_preconditions: List[List[Contract]], bases_have_func: bool,
                            preconditions: List[List[Contract]], func: Callable[..., Any]) -> List[List[Contract]]:
    """
    Collapse function preconditions with the preconditions collected from the base classes.

    :param base_preconditions: preconditions collected from the base classes (grouped by base class)
    :param bases_have_func: True if one of the base classes has the function
    :param preconditions: preconditions of the function (before the collapse)
    :param func: function whose preconditions we are collapsing
    :return: collapsed sequence of precondition groups
    """
    if not base_preconditions and bases_have_func and preconditions:
        raise TypeError(("The function {} can not weaken the preconditions because the bases specify "
                         "no preconditions at all. Hence this function must accept all possible input since "
                         "the preconditions are OR'ed and no precondition implies a dummy precondition which is always "
                         "fulfilled.").format(func.__qualname__))

    return base_preconditions + preconditions