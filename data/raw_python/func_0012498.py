def datetime_format(desired_format, datetime_instance=None,  *args, **kwargs):
    """
    Replaces format style phrases (listed in the dt_exps dictionary)
    with this datetime instance's information.

    .. code :: python

        reusables.datetime_format("Hey, it's {month-full} already!")
        "Hey, it's March already!"

    :param desired_format: string to add datetime details too
    :param datetime_instance: datetime.datetime instance, defaults to 'now'
    :param args: additional args to pass to str.format
    :param kwargs: additional kwargs to pass to str format
    :return: formatted string
    """
    for strf, exp in datetime_regex.datetime.format.items():
        desired_format = exp.sub(strf, desired_format)
    if not datetime_instance:
        datetime_instance = now()
    return datetime_instance.strftime(desired_format.format(*args, **kwargs))