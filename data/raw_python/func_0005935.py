def filter_locals(locals_dict, drop=None, include=None):
    """Filters a dictionary produced by locals().

    :param dict locals_dict:

    :param list drop: Keys to drop from dict.

    :param list include: Keys to include into dict.

    :rtype: dict
    """
    drop = drop or []
    drop.extend([
        'self',
        '__class__',  # py3
    ])

    include = include or locals_dict.keys()

    relevant_keys = [key for key in include if key not in drop]

    locals_dict = {k: v for k, v in locals_dict.items() if k in relevant_keys}

    return locals_dict