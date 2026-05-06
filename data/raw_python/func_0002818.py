def validateDayOfWeek(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, dayNames=ENGLISH_DAYS_OF_WEEK, excMsg=None):
    """Raises ValidationException if value is not a day of the week, such as 'Mon' or 'Friday'.
    Returns the titlecased day of the week.

    * value (str): The value being validated as a day of the week.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * dayNames (Mapping): A mapping of uppercase day abbreviations to day names, i.e. {'SUN': 'Sunday', ...} The default provides English day names.
    * excMsg (str): A custom message to use in the raised ValidationException.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateDayOfWeek('mon')
    'Monday'
    >>> pysv.validateDayOfWeek('THURSday')
    'Thursday'
    """

    # TODO - reuse validateChoice for this function

    # returns full day of the week str, e.g. 'Sunday'

    # Reuses validateMonth.
    try:
        return validateMonth(value, blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes, monthNames=ENGLISH_DAYS_OF_WEEK)
    except:
        # Replace the exception message.
        _raiseValidationException(_('%r is not a day of the week') % (_errstr(value)), excMsg)