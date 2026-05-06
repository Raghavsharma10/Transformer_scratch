def decorate_several_with_validation(func,
                                     _out_=None,         # type: ValidationFuncs
                                     none_policy=None,   # type: int
                                     **validation_funcs  # type: ValidationFuncs
                                     ):
    # type: (...) -> Callable
    """
    This method is equivalent to applying `decorate_with_validation` once for each of the provided arguments of
    the function `func` as well as output `_out_`. validation_funcs keyword arguments are validation functions for each
    arg name.

    Note that this method is less flexible than decorate_with_validation since
     * it does not allow to associate a custom error message or error type with each validation.
     * the none_policy is the same for all inputs and outputs

    :param func:
    :param _out_:
    :param validation_funcs:
    :param none_policy:
    :return: a function decorated with validation for all of the listed arguments and output if provided.
    """

    # add validation for output if provided
    if _out_ is not None:
        func = decorate_with_validation(func, _OUT_KEY, _out_, none_policy=none_policy)

    # add validation for each of the listed arguments
    for att_name, att_validation_funcs in validation_funcs.items():
        func = decorate_with_validation(func, att_name, att_validation_funcs, none_policy=none_policy)

    return func