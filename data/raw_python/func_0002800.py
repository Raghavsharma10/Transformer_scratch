def validateInt(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                min=None, max=None, lessThan=None, greaterThan=None, excMsg=None):
    """Raises ValidationException if value is not a int.

    Returns value, so it can be used inline in an expression:

        print(2 + validateInt(your_number))

    Note that since int() and ignore leading or trailing whitespace
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
    >>> pysv.validateInt('42')
    42
    >>> pysv.validateInt('forty two')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: 'forty two' is not an integer.
    """
    return validateNum(value=value, blank=blank, strip=strip, allowlistRegexes=None,
                       blocklistRegexes=blocklistRegexes, _numType='int', min=min, max=max,
                       lessThan=lessThan, greaterThan=greaterThan)