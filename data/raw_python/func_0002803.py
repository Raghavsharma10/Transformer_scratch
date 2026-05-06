def _validateParamsFor__validateToDateTimeFormat(formats, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None):
    """Raises PySimpleValidateException if the arguments are invalid. This is called by
    the validateTime() function to check its arguments. This code was
    refactored out to a separate function so that the PyInputPlus module (or
    other modules) could check their parameters' arguments for inputTime().
    """
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)
    if formats is None:
        raise PySimpleValidateException('formats parameter must be specified')

    if isinstance(formats, str):
        raise PySimpleValidateException('formats argument must be a non-str sequence of strftime format strings')

    try:
        len(formats)
    except:
        raise PySimpleValidateException('formats argument must be a non-str sequence of strftime format strings')

    for format in formats:
        try:
            time.strftime(format) # This will raise an exception if the format is invalid.
        except:
            raise PySimpleValidateException('formats argument contains invalid strftime format strings')