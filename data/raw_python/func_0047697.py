def process_args(mod_id, args, type_args):
    """
    Takes as input a list of arguments defined on a module and the information
    about the required arguments defined on the corresponding module type.
    Validates that the number of supplied arguments is valid and fills any
    missing arguments with their default values from the module type
    """
    res = list(args)
    if len(args) > len(type_args):
        raise ValueError(
            'Too many arguments specified for module "{}" (Got {}, expected '
            '{})'.format(mod_id, len(args), len(type_args))
        )
    for i in range(len(args), len(type_args)):
        arg_info = type_args[i]
        if "default" in arg_info:
            args.append(arg_info["default"])
        else:
            raise ValueError(
                'Not enough module arguments supplied for module "{}" (Got '
                '{}, expecting {})'.format(
                    mod_id, len(args), len(type_args)
                )
            )
    return args