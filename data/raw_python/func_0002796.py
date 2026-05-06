def _validateGenericParameters(blank, strip, allowlistRegexes, blocklistRegexes):
    """Returns None if the blank, strip, and blocklistRegexes parameters are valid
    of PySimpleValidate's validation functions have. Raises a PySimpleValidateException
    if any of the arguments are invalid."""

    # Check blank parameter.
    if not isinstance(blank, bool):
        raise PySimpleValidateException('blank argument must be a bool')

    # Check strip parameter.
    if not isinstance(strip, (bool, str, type(None))):
        raise PySimpleValidateException('strip argument must be a bool, None, or str')

    # Check allowlistRegexes parameter (including each regex in it).
    if allowlistRegexes is None:
        allowlistRegexes = [] # allowlistRegexes defaults to a blank list.

    try:
        len(allowlistRegexes) # Make sure allowlistRegexes is a sequence.
    except:
        raise PySimpleValidateException('allowlistRegexes must be a sequence of regex_strs')
    for response in allowlistRegexes:
        if not isinstance(response[0], str):
            raise PySimpleValidateException('allowlistRegexes must be a sequence of regex_strs')

    # Check allowlistRegexes parameter (including each regex in it).
    # NOTE: blocklistRegexes is NOT the same format as allowlistRegex, it can
    # include an "invalid input reason" string to display if the input matches
    # the blocklist regex.
    if blocklistRegexes is None:
        blocklistRegexes = [] # blocklistRegexes defaults to a blank list.

    try:
        len(blocklistRegexes) # Make sure blocklistRegexes is a sequence of (regex_str, str) or strs.
    except:
        raise PySimpleValidateException('blocklistRegexes must be a sequence of (regex_str, str) tuples or regex_strs')
    for response in blocklistRegexes:
        if isinstance(response, str):
            continue
        if len(response) != 2:
            raise PySimpleValidateException('blocklistRegexes must be a sequence of (regex_str, str) tuples or regex_strs')
        if not isinstance(response[0], str) or not isinstance(response[1], str):
            raise PySimpleValidateException('blocklistRegexes must be a sequence of (regex_str, str) tuples or regex_strs')