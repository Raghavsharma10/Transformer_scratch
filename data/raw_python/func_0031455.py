def condition2checker(condition):
    """Converts different condition types to callback"""
    if isinstance(condition, string_types):
        def smatcher(info):
            return fnmatch.fnmatch(info.filename, condition)

        return smatcher
    elif isinstance(condition, (list, tuple)) and isinstance(condition[0],
                                                             integer_types):
        def imatcher(info):
            return info.index in condition

        return imatcher
    elif callable(condition):
        return condition
    else:
        raise TypeError