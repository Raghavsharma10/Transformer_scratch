def validateRegexStr(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value can't be used as a regular expression string.
    Returns the value argument as a regex object.

    If you want to check if a string matches a regular expression, call
    validateRegex().

    * value (str): The value being validated as a regular expression string.
    * regex (str, regex): The regular expression to match the value against.
    * flags (int): Identical to the flags argument in re.compile(). Pass re.VERBOSE et al here.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateRegexStr('(cat)|(dog)')
    re.compile('(cat)|(dog)')
    >>> pysv.validateRegexStr('"(.*?)"')
    re.compile('"(.*?)"')
    >>> pysv.validateRegexStr('"(.*?"')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '"(.*?"' is not a valid regular expression: missing ), unterminated subpattern at position 1
    """

    # TODO - I'd be nice to check regexes in other languages, i.e. JS and Perl.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    try:
        return re.compile(value)
    except Exception as ex:
        _raiseValidationException(_('%r is not a valid regular expression: %s') % (_errstr(value), ex), excMsg)