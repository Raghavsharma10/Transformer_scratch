def on_all_(*validation_func):
    """
    Generates a validation_function for collection inputs where each element of the input will be validated against the
    validation_functions provided. For convenience, a list of validation_functions can be provided and will be replaced
    with an 'and_'.

    Note that if you want to apply DIFFERENT validation_functions for each element in the input, you should rather use
    on_each_.

    :param validation_func: the base validation function or list of base validation functions to use. A callable, a
    tuple(callable, help_msg_str), a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :return:
    """
    # create the validation functions
    validation_function_func = _process_validation_function_s(list(validation_func))

    def on_all_val(x):
        # validate all elements in x in turn
        for idx, x_elt in enumerate(x):
            try:
                res = validation_function_func(x_elt)
            except Exception as e:
                raise InvalidItemInSequence(wrong_value=x_elt, wrapped_func=validation_function_func, validation_outcome=e)

            if not result_is_success(res):
                # one element of x was not valid > raise
                # raise Failure('on_all_(' + str(validation_func) + '): failed validation for input '
                #                       'element [' + str(idx) + ']: ' + str(x_elt))
                raise InvalidItemInSequence(wrong_value=x_elt, wrapped_func=validation_function_func, validation_outcome=res)
        return True

    on_all_val.__name__ = 'apply_<{}>_on_all_elts'.format(get_callable_name(validation_function_func))
    return on_all_val