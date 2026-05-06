def validateNum(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, _numType='num',
                min=None, max=None, lessThan=None, greaterThan=None, excMsg=None):
    """Raises ValidationException if value is not a float or int.

    Returns value, so it can be used inline in an expression:

        print(2 + validateNum(your_number))

    Note that since int() and float() ignore leading or trailing whitespace
    when converting a string to a number, so does this validateNum().

    * value (str): The value being validated as an int or float.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * _numType (str): One of 'num', 'int', or 'float' for the kind of number to validate against, where 'num' means int or float.
    * min (int, float): The (inclusive) minimum value for the value to pass validation.
    * max (int, float): The (inclusive) maximum value for the value to pass validation.
    * lessThan (int, float): The (exclusive) minimum value for the value to pass validation.
    * greaterThan (int, float): The (exclusive) maximum value for the value to pass validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    If you specify min or max, you cannot also respectively specify lessThan
    or greaterThan. Doing so will raise PySimpleValidateException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateNum('3')
    3
    >>> pysv.validateNum('3.0')
    3.0
    >>> pysv.validateNum('    3.0    ')
    3.0
    >>> pysv.validateNum('549873259847598437598435798435793.589985743957435794357')
    5.498732598475984e+32
    >>> pysv.validateNum('4', lessThan=4)
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: Number must be less than 4.
    >>> pysv.validateNum('4', max=4)
    4
    >>> pysv.validateNum('4', min=2, max=5)
    4
    """

    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=None, blocklistRegexes=blocklistRegexes)
    _validateParamsFor_validateNum(min=min, max=max, lessThan=lessThan, greaterThan=greaterThan)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        # If we can convert value to an int/float, then do so. For example,
        # if an allowlist regex allows '42', then we should return 42/42.0.
        if (_numType == 'num' and '.' in value) or (_numType == 'float'):
            try:
                return float(value)
            except ValueError:
                return value # Return the value as is.
        if (_numType == 'num' and '.' not in value) or (_numType == 'int'):
            try:
                return int(value)
            except ValueError:
                return value # Return the value as is.

    # Validate the value's type (and convert value back to a number type).
    if (_numType == 'num' and '.' in value):
        # We are expecting a "num" (float or int) type and the user entered a float.
        try:
            value = float(value)
        except:
            _raiseValidationException(_('%r is not a number.') % (_errstr(value)), excMsg)
    elif (_numType == 'num' and '.' not in value):
        # We are expecting a "num" (float or int) type and the user entered an int.
        try:
            value = int(value)
        except:
            _raiseValidationException(_('%r is not a number.') % (_errstr(value)), excMsg)
    elif _numType == 'float':
        try:
            value = float(value)
        except:
            _raiseValidationException(_('%r is not a float.') % (_errstr(value)), excMsg)
    elif _numType == 'int':
        try:
            if float(value) % 1 != 0:
                # The number is a float that doesn't end with ".0"
                _raiseValidationException(_('%r is not an integer.') % (_errstr(value)), excMsg)
            value = int(float(value))
        except:
            _raiseValidationException(_('%r is not an integer.') % (_errstr(value)), excMsg)

    # Validate against min argument.
    if min is not None and value < min:
        _raiseValidationException(_('Number must be at minimum %s.') % (min), excMsg)

    # Validate against max argument.
    if max is not None and value > max:
        _raiseValidationException(_('Number must be at maximum %s.') % (max), excMsg)

    # Validate against max argument.
    if lessThan is not None and value >= lessThan:
        _raiseValidationException(_('Number must be less than %s.') % (lessThan), excMsg)

    # Validate against max argument.
    if greaterThan is not None and value <= greaterThan:
        _raiseValidationException(_('Number must be greater than %s.') % (greaterThan), excMsg)

    return value