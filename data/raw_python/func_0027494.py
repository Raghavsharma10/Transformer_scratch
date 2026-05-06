def or_(*validation_func  # type: ValidationFuncs
        ):
    # type: (...) -> Callable
    """
    An 'or' validator: returns `True` if at least one of the provided validators returns `True`. All exceptions will be
    silently caught. In case of failure, a global `AllValidatorsFailed` failure will be raised, together with details
    about all validation results.

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :return:
    """

    validation_func = _process_validation_function_s(list(validation_func), auto_and_wrapper=False)

    if len(validation_func) == 1:
        return validation_func[0]  # simplification for single validator case
    else:
        def or_v_(x):
            for validator in validation_func:
                # noinspection PyBroadException
                try:
                    res = validator(x)
                    if result_is_success(res):
                        # we can return : one validator was happy
                        return True
                except Exception:
                    # catch all silently
                    pass

            # no validator accepted: gather details and raise
            raise AllValidatorsFailed(validation_func, x)

        or_v_.__name__ = 'or({})'.format(get_callable_names(validation_func))
        return or_v_