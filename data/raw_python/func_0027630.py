def validate_out(*validation_func,  # type: ValidationFuncs
                 **kwargs):
    # type: (...) -> Callable
    """
    A decorator to apply function output validation to this function's output, with the provided base validation
    function(s). You may use several such decorators on a given function as long as they are stacked on top of each
    other (no external decorator in the middle)

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :param none_policy: describes how None values should be handled. See `NoneArgPolicy` for the various
    possibilities. Default is `NoneArgPolicy.ACCEPT_IF_OPTIONAl_ELSE_VALIDATE`.
    :return: a function decorator, able to transform a function into a function that will perform input validation
    before executing the function's code everytime it is executed.
    """
    def decorate(f):
        return decorate_with_validation(f, _OUT_KEY, *validation_func, **kwargs)
    return decorate