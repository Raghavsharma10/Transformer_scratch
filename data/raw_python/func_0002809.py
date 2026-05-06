def validateIP(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises ValidationException if value is not an IPv4 or IPv6 address.
    Returns the value argument.

    * value (str): The value being validated as an IP address.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateIP('127.0.0.1')
    '127.0.0.1'
    >>> pysv.validateIP('255.255.255.255')
    '255.255.255.255'
    >>> pysv.validateIP('256.256.256.256')
    Traceback (most recent call last):
    pysimplevalidate.ValidationException: '256.256.256.256' is not a valid IP address.
    >>> pysv.validateIP('1:2:3:4:5:6:7:8')
    '1:2:3:4:5:6:7:8'
    >>> pysv.validateIP('1::8')
    '1::8'
    >>> pysv.validateIP('fe80::7:8%eth0')
    'fe80::7:8%eth0'
    >>> pysv.validateIP('::255.255.255.255')
    '::255.255.255.255'
    """
    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    # Reuse the logic in validateRegex()
    try:
        try:
            # Check if value is an IPv4 address.
            if validateRegex(value=value, regex=IPV4_REGEX, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes):
                return value
        except:
            pass # Go on to check if it's an IPv6 address.

        # Check if value is an IPv6 address.
        if validateRegex(value=value, regex=IPV6_REGEX, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes):
            return value
    except ValidationException:
        _raiseValidationException(_('%r is not a valid IP address.') % (_errstr(value)), excMsg)