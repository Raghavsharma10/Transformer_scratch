def get_args(request_args, allowed_int_args=[], allowed_str_args=[]):
    """Check allowed argument names and return is as dictionary"""
    args = {}

    for allowed_int_arg in allowed_int_args:
        int_value = request_args.get(allowed_int_arg, default=None, type=None)
        if int_value:
            args[allowed_int_arg] = int(int_value)

    for allowed_str_arg in allowed_str_args:
        str_value = request_args.get(allowed_str_arg, default=None, type=None)
        if str_value:
            args[allowed_str_arg] = str_value

    return args