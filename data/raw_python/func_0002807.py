def validateFilename(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not a valid filename.
    Filenames can't contain \\ / : * ? " < > | or end with a space.
    Returns the value argument.

    Note that this validates filenames, not filepaths. The / and \\ characters
    are invalid for filenames.

    * value (str): The value being validated as an IP address.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateFilename('foobar.txt')
    'foobar.txt'
    >>> pysv.validateFilename('???.exe')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '???.exe' is not a valid filename.
    >>> pysv.validateFilename('/full/path/to/foo.txt')
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: '/full/path/to/foo.txt' is not a valid filename.
    """

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    if (value != value.strip()) or (any(c in value for c in '\\/:*?"<>|')):
        _raiseValidationException(_('%r is not a valid filename.') % (_errstr(value)), excMsg)
    return value