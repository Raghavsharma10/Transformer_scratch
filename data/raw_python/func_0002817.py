def validateMonth(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, monthNames=ENGLISH_MONTHS, excMsg=None):
    """Raises ValidationException if value is not a month, like 'Jan' or 'March'.
    Returns the titlecased month.

    * value (str): The value being validated as an email address.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * monthNames (Mapping): A mapping of uppercase month abbreviations to month names, i.e. {'JAN': 'January', ... }. The default provides English month names.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateMonth('Jan')
    'January'
    >>> pysv.validateMonth('MARCH')
    'March'
    """

    # returns full month name, e.g. 'January'

    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value


    try:
        if (monthNames == ENGLISH_MONTHS) and (1 <= int(value) <= 12): # This check here only applies to months, not when validateDayOfWeek() calls this function.
            return ENGLISH_MONTH_NAMES[int(value) - 1]
    except:
        pass # continue if the user didn't enter a number 1 to 12.

    # Both month names and month abbreviations will be at least 3 characters.
    if len(value) < 3:
        _raiseValidationException(_('%r is not a month.') % (_errstr(value)), excMsg)

    if value[:3].upper() in monthNames.keys(): # check if value is a month abbreviation
        return monthNames[value[:3].upper()] # It turns out that titlecase is good for all the month.
    elif value.upper() in monthNames.values(): # check if value is a month name
        return value.title()

    _raiseValidationException(_('%r is not a month.') % (_errstr(value)), excMsg)