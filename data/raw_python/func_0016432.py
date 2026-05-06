def match(fullname1, fullname2, strictness='default', options=None):
    """
    Takes two names and returns true if they describe the same person.

    :param string fullname1: first human name
    :param string fullname2: second human name
    :param string strictness: strictness settings to use
    :param dict options: custom strictness settings updates
    :return bool: the names match
    """

    if options is not None:
        settings = deepcopy(SETTINGS[strictness])
        deep_update_dict(settings, options)
    else:
        settings = SETTINGS[strictness]

    name1 = Name(fullname1)
    name2 = Name(fullname2)

    return name1.deep_compare(name2, settings)