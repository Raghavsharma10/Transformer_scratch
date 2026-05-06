def add_missing_optional_args_with_value_none(args, optional_args):
    '''
    Adds key-value pairs to the passed dictionary, so that
        afterwards, the dictionary can be used without needing
        to check for KeyErrors.

    If the keys passed as a second argument are not present,
        they are added with None as a value.

    :args: The dictionary to be completed.
    :optional_args: The keys that need to be added, if
        they are not present.
    :return: The modified dictionary.
    '''

    for name in optional_args:
        if not name in args.keys():
            args[name] = None
    return args