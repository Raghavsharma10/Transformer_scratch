def _validateParamsFor_validateChoice(choices, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                   numbered=False, lettered=False, caseSensitive=False, excMsg=None):
    """Raises PySimpleValidateException if the arguments are invalid. This is called by
    the validateChoice() function to check its arguments. This code was
    refactored out to a separate function so that the PyInputPlus module (or
    other modules) could check their parameters' arguments for inputChoice().
    """

    if not isinstance(caseSensitive, bool):
        raise PySimpleValidateException('caseSensitive argument must be a bool')

    try:
        len(choices)
    except:
        raise PySimpleValidateException('choices arg must be a sequence')
    if blank == False and len(choices) < 2:
        raise PySimpleValidateException('choices must have at least two items if blank is False')
    elif blank == True and len(choices) < 1:
        raise PySimpleValidateException('choices must have at least one item')


    _validateGenericParameters(blank=blank, strip=strip, allowlistRegexes=None, blocklistRegexes=blocklistRegexes)

    for choice in choices:
        if not isinstance(choice, str):
            raise PySimpleValidateException('choice %r must be a string' % (_errstr(choice)))
    if lettered and len(choices) > 26:
        raise PySimpleValidateException('lettered argument cannot be True if there are more than 26 choices')
    if numbered and lettered:
        raise PySimpleValidateException('numbered and lettered arguments cannot both be True')

    if len(choices) != len(set(choices)):
        raise PySimpleValidateException('duplicate entries in choices argument')

    if not caseSensitive and len(choices) != len(set([choice.upper() for choice in choices])):
        raise PySimpleValidateException('duplicate case-insensitive entries in choices argument')