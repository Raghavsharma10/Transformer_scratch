def validateURL(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not a URL.
    Returns the value argument.

    The "http" or "https" protocol part of the URL is optional.

    * value (str): The value being validated as a URL.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateURL('https://inventwithpython.com')
    'https://inventwithpython.com'
    >>> pysv.validateURL('inventwithpython.com')
    'inventwithpython.com'
    >>> pysv.validateURL('localhost')
    'localhost'
    >>> pysv.validateURL('mailto:al@inventwithpython.com')
    'mailto:al@inventwithpython.com'
    >>> pysv.validateURL('ftp://example.com')
    'example.com'
    >>> pysv.validateURL('https://inventwithpython.com/blog/2018/02/02/how-to-ask-for-programming-help/')
    'https://inventwithpython.com/blog/2018/02/02/how-to-ask-for-programming-help/'
    >>> pysv.validateURL('blah blah blah')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: 'blah blah blah' is not a valid URL.
    """

    # Reuse the logic in validateRegex()
    try:
        result = validateRegex(value=value, regex=URL_REGEX, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
        if result is not None:
            return result
    except ValidationException:
        # 'localhost' is also an acceptable URL:
        if value == 'localhost':
            return value

        _raiseValidationException(_('%r is not a valid URL.') % (value), excMsg)