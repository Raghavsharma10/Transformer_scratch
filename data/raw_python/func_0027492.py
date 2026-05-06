def and_(*validation_func  # type: ValidationFuncs
         ):
    # type: (...) -> Callable
    """
    An 'and' validator: it returns `True` if all of the provided validators return `True`, or raises a
    `AtLeastOneFailed` failure on the first `False` received or `Exception` caught.

    Note that an implicit `and_` is performed if you provide a list of validators to any of the entry points
    (`validate`, `validation`/`validator`, `@validate_arg`, `@validate_out`, `@validate_field` ...)

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :return:
    """

    validation_func = _process_validation_function_s(list(validation_func), auto_and_wrapper=False)

    if len(validation_func) == 1:
        return validation_func[0]  # simplification for single validator case: no wrapper
    else:
        def and_v_(x):
            for validator in validation_func:
                try:
                    res = validator(x)
                except Exception as e:
                    # one validator was unhappy > raise
                    raise AtLeastOneFailed(validation_func, x, cause=e)
                if not result_is_success(res):
                    # one validator was unhappy > raise
                    raise AtLeastOneFailed(validation_func, x)

            return True

        and_v_.__name__ = 'and({})'.format(get_callable_names(validation_func))
        return and_v_