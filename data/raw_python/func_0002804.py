def validateTime(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                 formats=('%H:%M:%S', '%H:%M', '%X'), excMsg=None):
    """Raises ValidationException if value is not a time formatted in one
    of the formats formats. Returns a datetime.time object of value.

    * value (str): The value being validated as a time.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * formats: A tuple of strings that can be passed to time.strftime, dictating the possible formats for a valid time.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateTime('12:00:01')
    datetime.time(12, 0, 1)
    >>> pysv.validateTime('13:00:01')
    datetime.time(13, 0, 1)
    >>> pysv.validateTime('25:00:01')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '25:00:01' is not a valid time.
    >>> pysv.validateTime('hour 12 minute 01', formats=['hour %H minute %M'])
    datetime.time(12, 1)
    """

    # TODO - handle this

    # Reuse the logic in _validateToDateTimeFormat() for this function.
    try:
        dt = _validateToDateTimeFormat(value, formats, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
        return datetime.time(dt.hour, dt.minute, dt.second, dt.microsecond)
    except ValidationException:
        _raiseValidationException(_('%r is not a valid time.') % (_errstr(value)), excMsg)