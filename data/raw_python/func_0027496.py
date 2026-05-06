def not_all(*validation_func,  # type: ValidationFuncs
            **kwargs
            ):
    # type: (...) -> Callable
    """
    An alias for not_(and_(validators)).

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
        tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
        are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
        `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
        of callables, they will be transformed to functions automatically.
    :param catch_all: an optional boolean flag. By default, only Failure are silently caught and turned into
        a 'ok' result. Turning this flag to True will assume that all exceptions should be caught and turned to a
        'ok' result
    :return:
    """
    catch_all = pop_kwargs(kwargs, [('catch_all', False)])

    # in case this is a list, create a 'and_' around it (otherwise and_ will return the validation function without
    # wrapping it)
    main_validator = and_(*validation_func)
    return not_(main_validator, catch_all=catch_all)