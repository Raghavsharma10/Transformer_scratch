def get_args_tuple(args, kwargs, arg_names, kwargs_defaults):
    """Generates a cache key from the passed in arguments."""
    args_list = list(args)
    args_len = len(args)
    all_args_len = len(arg_names)
    try:
        while args_len < all_args_len:
            arg_name = arg_names[args_len]
            if arg_name in kwargs_defaults:
                args_list.append(kwargs.get(arg_name, kwargs_defaults[arg_name]))
            else:
                args_list.append(kwargs[arg_name])
            args_len += 1
    except KeyError as e:
        raise TypeError("Missing argument %r" % (e.args[0],))
    return tuple(args_list)