def _collapse_postconditions(base_postconditions: List[Contract], postconditions: List[Contract]) -> List[Contract]:
    """
    Collapse function postconditions with the postconditions collected from the base classes.

    :param base_postconditions: postconditions collected from the base classes
    :param postconditions: postconditions of the function (before the collapse)
    :return: collapsed sequence of postconditions
    """
    return base_postconditions + postconditions