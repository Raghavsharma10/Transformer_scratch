def _number_of_line(member_tuple):
    """Try to return the number of the first line of the definition of a
    member of a module."""
    member = member_tuple[1]
    try:
        return member.__code__.co_firstlineno
    except AttributeError:
        pass
    try:
        return inspect.findsource(member)[1]
    except BaseException:
        pass
    for value in vars(member).values():
        try:
            return value.__code__.co_firstlineno
        except AttributeError:
            pass
    return 0