def validateEmail(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not an email address.
    Returns the value argument.

    * value (str): The value being validated as an email address.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateEmail('al@inventwithpython.com')
    'al@inventwithpython.com'
    >>> pysv.validateEmail('alinventwithpython.com')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: 'alinventwithpython.com' is not a valid email address.
    """

    # Reuse the logic in validateRegex()
    try:
        result = validateRegex(value=value, regex=EMAIL_REGEX, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
        if result is not None:
            return result
    except ValidationException:
        _raiseValidationException(_('%r is not a valid email address.') % (value), excMsg)