def validateBool(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, trueVal='True', falseVal='False', caseSensitive=False, excMsg=None):
    """Raises ValidationException if value is not an email address.
    Returns the yesVal or noVal argument, not value.

    * value (str): The value being validated as an email address.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
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

    # Replace the exception messages used in validateYesNo():
    trueVal = str(trueVal)
    falseVal = str(falseVal)
    if len(trueVal) == 0:
        raise PySimpleValidateException('trueVal argument must be a non-empty string.')
    if len(falseVal) == 0:
        raise PySimpleValidateException('falseVal argument must be a non-empty string.')
    if (trueVal == falseVal) or (not caseSensitive and trueVal.upper() == falseVal.upper()):
        raise PySimpleValidateException('trueVal and noVal arguments must be different.')
    if (trueVal[0] == falseVal[0]) or (not caseSensitive and trueVal[0].upper() == falseVal[0].upper()):
        raise PySimpleValidateException('first character of trueVal and noVal arguments must be different')

    result = validateYesNo(value, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes, yesVal=trueVal, noVal=falseVal, caseSensitive=caseSensitive, excMsg=None)

    # Return a bool value instead of a string.
    if result == trueVal:
        return True
    elif result == falseVal:
        return False
    else:
        assert False, 'inner validateYesNo() call returned something that was not yesVal or noVal. This should never happen.'