def validate_io(f=DECORATED,
                none_policy=None,      # type: int
                _out_=None,            # type: ValidationFuncs
                **kw_validation_funcs  # type: ValidationFuncs
                ):
    """
    A function decorator to add input validation prior to the function execution. It should be called with named
    arguments: for each function arg name, provide a single validation function or a list of validation functions to
    apply. If validation fails, it will raise an InputValidationError with details about the function, the input name,
    and any further information available from the validation function(s)

    For example:

    ```
    def is_even(x):
        return x % 2 == 0

    def gt(a):
        def gt(x):
            return x >= a
        return gt

    @validate_io(a=[is_even, gt(1)], b=is_even)
    def myfunc(a, b):
        print('hello')
    ```

    will generate the equivalent of :

    ```
    def myfunc(a, b):
        gt1 = gt(1)
        if (is_even(a) and gt1(a)) and is_even(b):
            print('hello')
        else:
            raise InputValidationError(...)
    ```

    :param none_policy: describes how None values should be handled. See `NoneArgPolicy` for the various
        possibilities. Default is `NoneArgPolicy.ACCEPT_IF_OPTIONAl_ELSE_VALIDATE`.
    :param _out_: a validation function or list of validation functions to apply to the function output. See
        kw_validation_funcs for details about the syntax.
    :param kw_validation_funcs: keyword arguments: for each of the function's input names, the validation function or
        list of validation functions to use. A validation function may be a callable, a tuple(callable, help_msg_str),
        a tuple(callable, failure_type), or a list of several such elements. Nested lists are supported and indicate an
        implicit `and_` (such as the main list). Tuples indicate an implicit `_failure_raiser`.
        [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead of callables, they
        will be transformed to functions automatically.
    :return: the decorated function, that will perform input validation before executing the function's code everytime
        it is executed.
    """
    return decorate_several_with_validation(f, none_policy=none_policy, _out_=_out_, **kw_validation_funcs)