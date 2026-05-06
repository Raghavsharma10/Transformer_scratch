def validateDatetime(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                     formats=('%Y/%m/%d %H:%M:%S', '%y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%m/%d/%y %H:%M:%S', '%x %H:%M:%S',
                              '%Y/%m/%d %H:%M', '%y/%m/%d %H:%M', '%m/%d/%Y %H:%M', '%m/%d/%y %H:%M', '%x %H:%M',
                              '%Y/%m/%d %H:%M:%S', '%y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%m/%d/%y %H:%M:%S', '%x %H:%M:%S'), excMsg=None):
    """Raises ValidationException if value is not a datetime formatted in one
    of the formats formats. Returns a datetime.datetime object of value.

    * value (str): The value being validated as a datetime.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * formats: A tuple of strings that can be passed to time.strftime, dictating the possible formats for a valid datetime.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateDatetime('2018/10/31 12:00:01')
    datetime.datetime(2018, 10, 31, 12, 0, 1)
    >>> pysv.validateDatetime('10/31/2018 12:00:01')
    datetime.datetime(2018, 10, 31, 12, 0, 1)
    >>> pysv.validateDatetime('10/31/2018')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '10/31/2018' is not a valid date and time.
    """

    # Reuse the logic in _validateToDateTimeFormat() for this function.
    try:
        return _validateToDateTimeFormat(value, formats, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
    except ValidationException:
        _raiseValidationException(_('%r is not a valid date and time.') % (_errstr(value)), excMsg)