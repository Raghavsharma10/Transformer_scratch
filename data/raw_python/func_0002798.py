def validateStr(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not a string. This function
    is identical to the built-in input() function, but also offers the
    PySimpleValidate features of not allowing blank values by default,
    automatically stripping whitespace, and having allowlist/blocklist
    regular expressions.

    Returns value, so it can be used inline in an expression:

        print('Hello, ' + validateStr(your_name))

    * value (str): The value being validated as a string.
    * blank (bool): If True, a blank string will be accepted. Defaults to False. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateStr('hello')
    'hello'
    >>> pysv.validateStr('')
    Traceback (most recent call last):
      ...
    pysimplevalidate.ValidationException: Blank values are not allowed.
    >>> pysv.validateStr('', blank=True)
    ''
    >>> pysv.validateStr('    hello    ')
    'hello'
    >>> pysv.validateStr('hello', blocklistRegexes=['hello'])
    Traceback (most recent call last):
      ...
    pysimplevalidate.ValidationException: This response is invalid.
    >>> pysv.validateStr('hello', blocklistRegexes=[('hello', 'Hello is not allowed')])
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: Hello is not allowed
    >>> pysv.validateStr('hello', allowlistRegexes=['hello'], blocklistRegexes=['llo'])
    'hello'
    """

    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=None, blocklistRegexes=blocklistRegexes)
    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)

    return value