def failure_raiser(*validation_func,   # type: ValidationFuncs
                   **kwargs
                   ):
    # type: (...) -> Callable
    """
    This function is automatically used if you provide a tuple `(<function>, <msg>_or_<Failure_type>)`, to any of the
    methods in this page or to one of the `valid8` decorators. It transforms the provided `<function>` into a failure
    raiser, raising a subclass of `Failure` in case of failure (either not returning `True` or raising an exception)

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :param failure_type: a subclass of `WrappingFailure` that should be raised in case of failure
    :param help_msg: a string help message for the raised `WrappingFailure`. Optional (default = WrappingFailure with
    no help message).
    :param kw_context_args
    :return:
    """
    failure_type, help_msg = pop_kwargs(kwargs, [('failure_type', None), ('help_msg', None)], allow_others=True)
    # the rest of keyword arguments is used as context.
    kw_context_args = kwargs

    main_func = _process_validation_function_s(list(validation_func))
    return _failure_raiser(main_func, failure_type=failure_type,  help_msg=help_msg, **kw_context_args)