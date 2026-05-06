def validate_arg(f,
                 arg_name,
                 *validation_func,  # type: ValidationFuncs
                 **kwargs
                 ):
    # type: (...) -> Callable
    """
    A decorator to apply function input validation for the given argument name, with the provided base validation
    function(s). You may use several such decorators on a given function as long as they are stacked on top of each
    other (no external decorator in the middle)

    :param arg_name:
    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
        tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
        are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
        `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
        of callables, they will be transformed to functions automatically.
    :param error_type: a subclass of ValidationError to raise in case of validation failure. By default a
        ValidationError will be raised with the provided help_msg
    :param help_msg: an optional help message to be used in the raised error in case of validation failure.
    :param none_policy: describes how None values should be handled. See `NoneArgPolicy` for the various
        possibilities. Default is `NoneArgPolicy.ACCEPT_IF_OPTIONAl_ELSE_VALIDATE`.
    :param kw_context_args: optional contextual information to store in the exception, and that may be also used
        to format the help message
    :return: a function decorator, able to transform a function into a function that will perform input validation
        before executing the function's code everytime it is executed.
    """
    return decorate_with_validation(f, arg_name, *validation_func, **kwargs)