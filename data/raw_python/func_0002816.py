def validateState(value, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None, excMsg=None, returnStateName=False):
    """Raises ValidationException if value is not a USA state.
    Returns the capitalized state abbreviation, unless returnStateName is True
    in which case it returns the titlecased state name.

    * value (str): The value being validated as an email address.
    * blank (bool):  If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * excMsg (str): A custom message to use in the raised ValidationException.
    * returnStateName (bool): If True, the full state name is returned, i.e. 'California'. Otherwise, the abbreviation, i.e. 'CA'. Defaults to False.

    >>> import pysimplevalidate as pysv
    >>> pysv.validateState('tx')
    'TX'
    >>> pysv.validateState('california')
    'CA'
    >>> pysv.validateState('WASHINGTON')
    'WA'
    >>> pysv.validateState('WASHINGTON', returnStateName=True)
    'Washington'
    """

    # TODO - note that this is USA-centric. I should work on trying to make this more international.

    # Validate parameters.
    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=allowlistRegexes, blocklistRegexes=blocklistRegexes)

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    if value.upper() in USA_STATES_UPPER.keys(): # check if value is a state abbreviation
        if returnStateName:
            return USA_STATES[value.upper()] # Return full state name.
        else:
            return value.upper() # Return abbreviation.
    elif value.title() in USA_STATES.values(): # check if value is a state name
        if returnStateName:
            return value.title() # Return full state name.
        else:
            return USA_STATES_REVERSED[value.title()] # Return abbreviation.

    _raiseValidationException(_('%r is not a state.') % (_errstr(value)), excMsg)