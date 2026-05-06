def _process_validation_function_s(validation_func,       # type: ValidationFuncs
                                   auto_and_wrapper=True  # type: bool
                                   ):
    # type: (...) -> Union[Callable, List[Callable]]
    """
    This function handles the various ways that users may enter 'validation functions', so as to output a single
    callable method. Setting "auto_and_wrapper" to False allows callers to get a list of callables instead.

    valid8 supports the following expressions for 'validation functions'
     * <ValidationFunc>
     * List[<ValidationFunc>(s)]. The list must not be empty.

    <ValidationFunc> may either be
     * a callable or a mini-lambda expression (instance of LambdaExpression - in which case it is automatically
     'closed').
     * a Tuple[callable or mini-lambda expression ; failure_type]. Where failure type should be a subclass of
     valid8.Failure. In which case the tuple will be replaced with a _failure_raiser(callable, failure_type)

    When the contents provided does not match the above, this function raises a ValueError. Otherwise it produces a
    list of callables, that will typically be turned into a `and_` in the nominal case except if this is called inside
    `or_` or `xor_`.

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
        tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
        are supported and indicate an implicit `and_`. Tuples indicate an implicit
        `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
        of callables, they will be transformed to functions automatically.
    :param auto_and_wrapper: if True (default), this function returns a single callable that is a and_() of all
        functions. Otherwise a list is returned.
    :return:
    """

    # handle the case where validation_func is not yet a list or is empty or none
    if validation_func is None:
        raise ValueError('mandatory validation_func is None')

    elif not isinstance(validation_func, list):
        # so not use list() because we do not want to convert tuples here.
        validation_func = [validation_func]

    elif len(validation_func) == 0:
        raise ValueError('provided validation_func list is empty')

    # now validation_func is a non-empty list
    final_list = []
    for v in validation_func:
        # special case of a LambdaExpression: automatically convert to a function
        # note: we have to do it before anything else (such as .index) otherwise we may get failures
        v = as_function(v)

        if isinstance(v, tuple):
            # convert all the tuples to failure raisers
            if len(v) == 2:
                if isinstance(v[1], str):
                    final_list.append(_failure_raiser(v[0], help_msg=v[1]))
                elif isinstance(v[1], type) and issubclass(v[1], WrappingFailure):
                    final_list.append(_failure_raiser(v[0], failure_type=v[1]))
                else:
                    raise TypeError('base validation function(s) not compliant with the allowed syntax. Base validation'
                                    ' function(s) can be {}. Found [{}].'.format(supported_syntax, str(v)))
            else:
                raise TypeError('base validation function(s) not compliant with the allowed syntax. Base validation'
                                ' function(s) can be {}. Found [{}].'.format(supported_syntax, str(v)))

        elif callable(v):
            # use the validator directly
            final_list.append(v)

        elif isinstance(v, list):
            # a list is an implicit and_, make it explicit
            final_list.append(and_(*v))

        else:
            raise TypeError('base validation function(s) not compliant with the allowed syntax. Base validation'
                            ' function(s) can be {}. Found [{}].'.format(supported_syntax, str(v)))

    # return what is required:
    if auto_and_wrapper:
        # a single callable doing the 'and'
        return and_(*final_list)
    else:
        # or the list (typically for use inside or_(), xor_()...)
        return final_list