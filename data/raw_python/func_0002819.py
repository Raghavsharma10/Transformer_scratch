def validateDayOfMonth(value, year, month, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not a day of the month, from
    1 to 28, 29, 30, or 31 depending on the month and year.
    Returns value.

    * value (str): The value being validated as existing as a numbered day in the given year and month.
    * year (int): The given year.
    * month (int): The given month. 1 is January, 2 is February, and so on.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateDayOfMonth('31', 2019, 10)
    31
    >>> pysv.validateDayOfMonth('32', 2019, 10)
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '32' is not a day in the month of October 2019
    >>> pysv.validateDayOfMonth('29', 2004, 2)
    29
    >>> pysv.validateDayOfMonth('29', 2005, 2)
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '29' is not a day in the month of February 2005

    """
    try:
        daysInMonth = calendar.monthrange(year, month)[1]
    except:
        raise PySimpleValidateException('invalid arguments for year and/or month')

    try:
        return validateInt(value, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes, min=1, max=daysInMonth)
    except:
        # Replace the exception message.
        _raiseValidationException(_('%r is not a day in the month of %s %s') % (_errstr(value), ENGLISH_MONTH_NAMES[month - 1], year), excMsg)