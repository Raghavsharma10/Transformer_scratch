def validateYesNo(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, yesVal='yes', noVal='no', caseSensitive=False, excMsg=None):
    """Raises ValidationException if value is not a yes or no response.
    Returns the yesVal or noVal argument, not value.

    Note that value can be any case (by default) and can also just match the

    * value (str): The value being validated as an email address.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * caseSensitive (bool): Determines if value must match the case of yesVal and noVal. Defaults to False.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateYesNo('y')
    'yes'
    >>> pysv.validateYesNo('YES')
    'yes'
    >>> pysv.validateYesNo('No')
    'no'
    >>> pysv.validateYesNo('OUI', yesVal='oui', noVal='no')
    'oui'
    """

    # Validate parameters. TODO - can probably improve this to remove the duplication.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    yesVal = str(yesVal)
    noVal = str(noVal)
    if len(yesVal) == 0:
        raise PySimpleValidateException('yesVal argument must be a non-empty string.')
    if len(noVal) == 0:
        raise PySimpleValidateException('noVal argument must be a non-empty string.')
    if (yesVal == noVal) or (not caseSensitive and yesVal.upper() == noVal.upper()):
        raise PySimpleValidateException('yesVal and noVal arguments must be different.')
    if (yesVal[0] == noVal[0]) or (not caseSensitive and yesVal[0].upper() == noVal[0].upper()):
        raise PySimpleValidateException('first character of yesVal and noVal arguments must be different')

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    if caseSensitive:
        if value in (yesVal, yesVal[0]):
            return yesVal
        elif value in (noVal, noVal[0]):
            return noVal
    else:
        if value.upper() in (yesVal.upper(), yesVal[0].upper()):
            return yesVal
        elif value.upper() in (noVal.upper(), noVal[0].upper()):
            return noVal

    _raiseValidationException(_('%r is not a valid %s/%s response.') % (_errstr(value), yesVal, noVal), excMsg)