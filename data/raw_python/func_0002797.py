def _validateParamsFor_validateNum(min=None, max=None, lessThan=None, greaterThan=None):
    """Raises an exception if the arguments are invalid. This is called by
    the validateNum(), validateInt(), and validateFloat() functions to
    check its arguments. This code was refactored out to a separate function
    so that the PyInputPlus module (or other modules) could check their
    parameters' arguments for inputNum() etc.
    """

    if (min is not None) and (greaterThan is not None):
        raise PySimpleValidateException('only one argument for min or greaterThan can be passed, not both')
    if (max is not None) and (lessThan is not None):
        raise PySimpleValidateException('only one argument for max or lessThan can be passed, not both')

    if (min is not None) and (max is not None) and (min > max):
        raise PySimpleValidateException('the min argument must be less than or equal to the max argument')
    if (min is not None) and (lessThan is not None) and (min >= lessThan):
        raise PySimpleValidateException('the min argument must be less than the lessThan argument')
    if (max is not None) and (greaterThan is not None) and (max <= greaterThan):
        raise PySimpleValidateException('the max argument must be greater than the greaterThan argument')

    for name, val in (('min', min), ('max', max),
                      ('lessThan', lessThan), ('greaterThan', greaterThan)):
        if not isinstance(val, (int, float, type(None))):
            raise PySimpleValidateException(name + ' argument must be int, float, or NoneType')