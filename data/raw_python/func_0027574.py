def is_valid(value,
             *validation_func,  # type: Union[Callable, List[Callable]]
             **kwargs
             ):
    # type: (...) -> bool
    """
    Validates value `value` using validation function(s) `validator_func`.
    As opposed to `assert_valid`, this function returns a boolean indicating if validation was a success or a failure.
    It is therefore designed to be used within if ... else ... statements:

    ```python
    if is_valid(x, isfinite):
        ...<code>
    else
        ...<code>
    ```

    Note: this is a friendly alias for `return _validator(validator_func, return_bool=True)(value)`

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :param value: the value to validate
    :param none_policy: describes how None values should be handled. See `NonePolicy` for the various possibilities.
    Default is `NonePolicy.VALIDATE`, meaning that None values will be treated exactly like other values and follow
    the same validation process. Note that errors raised by NonePolicy.FAIL will be caught and transformed into a
    returned value of False
    :return: True if validation was a success, False otherwise
    """
    none_policy = pop_kwargs(kwargs, [('none_policy', None)])

    return Validator(*validation_func, none_policy=none_policy).is_valid(value)