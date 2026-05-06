def assert_valid(name,              # type: str
                 value,             # type: Any
                 *validation_func,  # type: ValidationFuncs
                 **kwargs):
    """
    Validates value `value` using validation function(s) `base_validator_s`.
    As opposed to `is_valid`, this function raises a `ValidationError` if validation fails.

    It is therefore designed to be used for defensive programming, in an independent statement before the code that you
    intent to protect.

    ```python
    assert_valid(x, isfinite):
    ...<your code>
    ```

    Note: this is a friendly alias for `_validator(base_validator_s)(value)`

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
        tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
        are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
        `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
        of callables, they will be transformed to functions automatically.
    :param name: the name of the variable to validate. It will be used in error messages
    :param value: the value to validate
    :param none_policy: describes how None values should be handled. See `NonePolicy` for the various possibilities.
        Default is `NonePolicy.VALIDATE`, meaning that None values will be treated exactly like other values and follow
        the same validation process.
    :param error_type: a subclass of ValidationError to raise in case of validation failure. By default a
        ValidationError will be raised with the provided help_msg
    :param help_msg: an optional help message to be used in the raised error in case of validation failure.
    :param kw_context_args: optional keyword arguments providing additional context, that will be provided to the error
        in case of validation failure
    :return: nothing in case of success. In case of failure, raises a <error_type> if provided, or a ValidationError.
    """
    error_type, help_msg, none_policy = pop_kwargs(kwargs, [('error_type', None),
                                                            ('help_msg', None),
                                                            ('none_policy', None)], allow_others=True)
    # the rest of keyword arguments is used as context.
    kw_context_args = kwargs

    return Validator(*validation_func, error_type=error_type, help_msg=help_msg,
                     none_policy=none_policy).assert_valid(name=name, value=value, **kw_context_args)