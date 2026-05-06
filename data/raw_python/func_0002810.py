def validateRegex(value, regex, flags=0, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value does not match the regular expression in regex.
    Returns the value argument.

    This is similar to calling inputStr() and using the allowlistRegexes
    keyword argument, however, validateRegex() allows you to pass regex
    flags such as re.IGNORECASE or re.VERBOSE. You can also pass a regex
    object directly.

    If you want to check if a string is a regular expression string, call
    validateRegexStr().

    * value (str): The value being validated as a regular expression string.
    * regex (str, regex): The regular expression to match the value against.
    * flags (int): Identical to the flags argument in re.compile(). Pass re.VERBOSE et al here.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> pysv.validateRegex('cat bat rat', r'(cat)|(dog)|(moose)', re.IGNORECASE)
    'cat'
    >>> pysv.validateRegex('He said "Hello".', r'"(.*?)"', re.IGNORECASE)
    '"Hello"'
    """

    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    # Search value with regex, whether regex is a str or regex object.
    if isinstance(regex, str):
        # TODO - check flags to see they're valid regex flags.
        mo = re.compile(regex, flags).search(value)
    elif isinstance(regex, REGEX_TYPE):
        mo = regex.search(value)
    else:
        raise PySimpleValidateException('regex must be a str or regex object')

    if mo is not None:
        return mo.group()
    else:
        _raiseValidationException(_('%r does not match the specified pattern.') % (_errstr(value)), excMsg)