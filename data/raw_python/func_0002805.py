def validateDate(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                 formats=('%Y/%m/%d', '%y/%m/%d', '%m/%d/%Y', '%m/%d/%y', '%x'), excMsg=None):
    """Raises ValidationException if value is not a time formatted in one
    of the formats formats. Returns a datetime.date object of value.

    * value (str): The value being validated as a time.
    * blank (bool): If True, a blank string for value will be accepted.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * formats: A tuple of strings that can be passed to time.strftime, dictating the possible formats for a valid date.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateDate('2/29/2004')
    datetime.date(2004, 2, 29)
    >>> pysv.validateDate('2/29/2005')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '2/29/2005' is not a valid date.
    >>> pysv.validateDate('September 2019', formats=['%B %Y'])
    datetime.date(2019, 9, 1)
    """
    # Reuse the logic in _validateToDateTimeFormat() for this function.
    try:
        dt = _validateToDateTimeFormat(value, formats, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
        return datetime.date(dt.year, dt.month, dt.day)
    except ValidationException:
        _raiseValidationException(_('%r is not a valid date.') % (_errstr(value)), excMsg)