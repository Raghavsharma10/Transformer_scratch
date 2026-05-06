def move_if_not_on_dow(original, replacement, dow_not_orig, dow_replacement):
    """
    Return a lambda function.

    Lambda checks that either the original day does not fall on a given
    weekday, or that the replacement day does fall on the expected weekday.
    """
    return lambda x: (
        (x.hdate.day == original and x.gdate.weekday() != dow_not_orig) or
        (x.hdate.day == replacement and x.gdate.weekday() == dow_replacement))