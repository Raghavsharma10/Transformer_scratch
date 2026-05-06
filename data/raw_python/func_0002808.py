def validateFilepath(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None, mustExist=False):
    r"""Raises ValidationException if value is not a valid filename.
    Filenames can't contain \\ / : * ? " < > |
    Returns the value argument.

    * value (str): The value being validated as an IP address.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateFilepath('foo.txt')
    'foo.txt'
    >>> pysv.validateFilepath('/spam/foo.txt')
    '/spam/foo.txt'
    >>> pysv.validateFilepath(r'c:\spam\foo.txt')
    'c:\\spam\\foo.txt'
    >>> pysv.validateFilepath(r'c:\spam\???.txt')
    Traceback (most recent call last):
      ...
    pysimplevalidate.ValidationException: 'c:\\spam\\???.txt' is not a valid file path.
    """
    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    if (value != value.strip()) or (any(c in value for c in '*?"<>|')): # Same as validateFilename, except we allow \ and / and :
        if ':' in value:
            if value.find(':', 2) != -1 or not value[0].isalpha():
                # For Windows: Colon can only be found at the beginning, e.g. 'C:\', or the first letter is not a letter drive.
                _raiseValidationException(_('%r is not a valid file path.') % (_errstr(value)), excMsg)
        _raiseValidationException(_('%r is not a valid file path.') % (_errstr(value)), excMsg)
    return value
    raise NotImplementedError()