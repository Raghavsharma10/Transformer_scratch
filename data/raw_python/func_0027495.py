def xor_(*validation_func  # type: ValidationFuncs
         ):
    # type: (...) -> Callable
    """
    A 'xor' validation function: returns `True` if exactly one of the provided validators returns `True`. All exceptions
    will be silently caught. In case of failure, a global `XorTooManySuccess` or `AllValidatorsFailed` will be raised,
    together with details about the various validation results.

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :return:
    """

    validation_func = _process_validation_function_s(list(validation_func), auto_and_wrapper=False)

    if len(validation_func) == 1:
        return validation_func[0]  # simplification for single validation function case
    else:
        def xor_v_(x):
            ok_validators = []
            for val_func in validation_func:
                # noinspection PyBroadException
                try:
                    res = val_func(x)
                    if result_is_success(res):
                        ok_validators.append(val_func)
                except Exception:
                    pass

            # return if were happy or not
            if len(ok_validators) == 1:
                # one unique validation function happy: success
                return True

            elif len(ok_validators) > 1:
                # several validation_func happy : fail
                raise XorTooManySuccess(validation_func, x)

            else:
                # no validation function happy, fail
                raise AllValidatorsFailed(validation_func, x)

        xor_v_.__name__ = 'xor({})'.format(get_callable_names(validation_func))
        return xor_v_