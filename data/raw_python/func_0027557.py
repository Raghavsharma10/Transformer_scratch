def on_each_(*validation_functions_collection):
    """
    Generates a validation_function for collection inputs where each element of the input will be validated against the
    corresponding validation_function(s) in the validation_functions_collection. Validators inside the tuple can be
    provided as a list for convenience, this will be replaced with an 'and_' operator if the list has more than one
    element.

    Note that if you want to apply the SAME validation_functions to all elements in the input, you should rather use
    on_all_.

    :param validation_functions_collection: a sequence of (base validation function or list of base validation functions
    to use).
    A base validation function may be a callable, a tuple(callable, help_msg_str), a tuple(callable, failure_type), or
    a list of several such elements. Nested lists are supported and indicate an implicit `and_` (such as the main list).
    Tuples indicate an implicit `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/)
    expressions can be used instead of callables, they will be transformed to functions automatically.
    :return:
    """
    # create a tuple of validation functions.
    validation_function_funcs = tuple(_process_validation_function_s(validation_func)
                                      for validation_func in validation_functions_collection)

    # generate a validation function based on the tuple of validation_functions lists
    def on_each_val(x  # type: Tuple
                    ):
        if len(validation_function_funcs) != len(x):
            raise Failure('on_each_: x does not have the same number of elements than validation_functions_collection.')
        else:
            # apply each validation_function on the input with the same position in the collection
            idx = -1
            for elt, validation_function_func in zip(x, validation_function_funcs):
                idx += 1
                try:
                    res = validation_function_func(elt)
                except Exception as e:
                    raise InvalidItemInSequence(wrong_value=elt,
                                                wrapped_func=validation_function_func,
                                                validation_outcome=e)

                if not result_is_success(res):
                    # one validation_function was unhappy > raise
                    # raise Failure('on_each_(' + str(validation_functions_collection) + '): _validation_function [' + str(idx)
                    #               + '] (' + str(validation_functions_collection[idx]) + ') failed validation for '
                    #                       'input ' + str(x[idx]))
                    raise InvalidItemInSequence(wrong_value=elt,
                                                wrapped_func=validation_function_func,
                                                validation_outcome=res)
            return True

    on_each_val.__name__ = 'map_<{}>_on_elts' \
                           ''.format('(' + ', '.join([get_callable_name(f) for f in validation_function_funcs]) + ')')
    return on_each_val