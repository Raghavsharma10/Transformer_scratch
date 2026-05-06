def normrelpath(base, target):
    """
    This function takes the base and target arguments as paths, and
    returns an equivalent relative path from base to the target, if both
    provided paths are absolute.
    """

    if not all(map(isabs, [base, target])):
        return target

    return relpath(normpath(target), dirname(normpath(base)))