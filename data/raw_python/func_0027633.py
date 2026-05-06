def decorate_with_validation(func,
                             arg_name,          # type: str
                             *validation_func,  # type: ValidationFuncs
                             **kwargs):
    # type: (...) -> Callable
    """
    This method is the inner method used in `@validate_io`, `@validate_arg` and `@validate_out`.
    It can be used if you with to perform decoration manually without a decorator.

    :param func:
    :param arg_name: the name of the argument to validate or _OUT_KEY for output validation
    :param validation_func: the validation function or
        list of validation functions to use. A validation function may be a callable, a tuple(callable, help_msg_str),
        a tuple(callable, failure_type), or a list of several such elements. Nested lists are supported and indicate an
        implicit `and_` (such as the main list). Tuples indicate an implicit `_failure_raiser`.
        [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead of callables, they
        will be transformed to functions automatically.
    :param error_type: a subclass of ValidationError to raise in case of validation failure. By default a
        ValidationError will be raised with the provided help_msg
    :param help_msg: an optional help message to be used in the raised error in case of validation failure.
    :param none_policy: describes how None values should be handled. See `NoneArgPolicy` for the various possibilities.
        Default is `NoneArgPolicy.ACCEPT_IF_OPTIONAl_ELSE_REJECT`.
    :param kw_context_args: optional contextual information to store in the exception, and that may be also used
        to format the help message
    :return: the decorated function, that will perform input validation (using `_assert_input_is_valid`) before
        executing the function's code everytime it is executed.
    """
    error_type, help_msg, none_policy, _constructor_of_cls_ = pop_kwargs(kwargs, [('error_type', None),
                                                                                  ('help_msg', None),
                                                                                  ('none_policy', None),
                                                                                  ('_constructor_of_cls_', None)],
                                                                         allow_others=True)
    # the rest of keyword arguments is used as context.
    kw_context_args = kwargs

    none_policy = none_policy or NoneArgPolicy.SKIP_IF_NONABLE_ELSE_VALIDATE

    # retrieve target function signature
    func_sig = signature(func)

    # create the new validator
    if _constructor_of_cls_ is None:
        # standard method: input validator
        new_validator = _create_function_validator(func, func_sig, arg_name, *validation_func,
                                                   none_policy=none_policy, error_type=error_type,
                                                   help_msg=help_msg, **kw_context_args)
    else:
        # class constructor: field validator
        new_validator = _create_function_validator(func, func_sig, arg_name, *validation_func,
                                                   none_policy=none_policy, error_type=error_type,
                                                   help_msg=help_msg, validated_class=_constructor_of_cls_,
                                                   validated_class_field_name=arg_name,
                                                   **kw_context_args)

    # decorate or update decorator with this new validator
    return decorate_with_validators(func, func_signature=func_sig, **{arg_name: new_validator})