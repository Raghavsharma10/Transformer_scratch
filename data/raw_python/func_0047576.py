def class_check_para(**kw):
    """
    force check accept and return,
    decorator, @class_check_para(accept=, returns=, mail=)
    :param kw:
    :return:
    """
    try:
        def decorator(f):
            def new_f(*args):
                if "accepts" in kw:
                    assert len(args) == len(kw["accepts"]) + 1
                    arg_types = tuple(map(type, args[1:]))
                    if arg_types != kw["accepts"]:
                        msg = decorator_info(f.__name__, kw["accepts"],
                                             arg_types, 0)
                        print('TypeWarning: ', msg)
                        raise TypeError(msg)
                result = f(*args)
                if "returns" in kw:
                    res_type = type(result)
                    if res_type != kw["returns"]:
                        msg = decorator_info(f.__name__, (kw["returns"],),
                                             (res_type,), 1)
                        print('TypeWarning: ', msg)
                        raise TypeError(msg)
                return result

            new_f.__name__ = f.__name__
            return new_f

        return decorator
    except KeyError as ke:
        raise KeyError(ke.message + "is not a valid keyword argument")
    except TypeError as te:
        raise TypeError(te.message)