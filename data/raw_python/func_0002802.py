def validateChoice(value, choices, blank=False, strip=None, allowlistRegexes=None, blocklistRegexes=None,
                   numbered=False, lettered=False, caseSensitive=False, excMsg=None):
    """Raises ValidationException if value is not one of the values in
    choices. Returns the selected choice.

    Returns the value in choices that was selected, so it can be used inline
    in an expression:

        print('You chose ' + validateChoice(your_choice, ['cat', 'dog']))

    Note that value itself is not returned: validateChoice('CAT', ['cat', 'dog'])
    will return 'cat', not 'CAT'.

    If lettered is True, lower or uppercase letters will be accepted regardless
    of what caseSensitive is set to. The caseSensitive argument only matters
    for matching with the text of the strings in choices.

    * value (str): The value being validated.
    * blank (bool): If True, a blank string will be accepted. Defaults to False.
    * strip (bool, str, None): If None, whitespace is stripped from value. If a str, the characters in it are stripped from value. If False, nothing is stripped.
    * allowlistRegexes (Sequence, None): A sequence of regex str that will explicitly pass validation, even if they aren't numbers.
    * blocklistRegexes (Sequence, None): A sequence of regex str or (regex_str, response_str) tuples that, if matched, will explicitly fail validation.
    * numbered (bool): If True, this function will also accept a string of the choice's number, i.e. '1' or '2'.
    * lettered (bool): If True, this function will also accept a string of the choice's letter, i.e. 'A' or 'B' or 'a' or 'b'.
    * caseSensitive (bool): If True, then the exact case of the option must be entered.
    * excMsg (str): A custom message to use in the raised ValidationException.

    Returns the choice selected as it appeared in choices. That is, if 'cat'
    was a choice and the user entered 'CAT' while caseSensitive is False,
    this function will return 'cat'.


    >>> import pysimplevalidate as pysv
    >>> pysv.validateChoice('dog', ['dog', 'cat', 'moose'])
    'dog'

    >>> pysv.validateChoice('DOG', ['dog', 'cat', 'moose'])
    'dog'

    >>> pysv.validateChoice('2', ['dog', 'cat', 'moose'], numbered=True)
    'cat'

    >>> pysv.validateChoice('a', ['dog', 'cat', 'moose'], lettered=True)
    'dog'

    >>> pysv.validateChoice('C', ['dog', 'cat', 'moose'], lettered=True)
    'moose'

    >>> pysv.validateChoice('dog', ['dog', 'cat', 'moose'], lettered=True)
    'dog'

    >>> pysv.validateChoice('spider', ['dog', 'cat', 'moose'])
    Traceback (most recent call last):
        ...
    pysimplevalidate.ValidationException: 'spider' is not a valid choice.
    """

    # Validate parameters.
    _validateParamsFor_validateChoice(choices=choices, blank=blank, strip=strip, allowlistRegexes=None,
        blocklistRegexes=blocklistRegexes, numbered=numbered, lettered=lettered, caseSensitive=caseSensitive)

    if '' in choices:
        # blank needs to be set to True here, otherwise '' won't be accepted as a choice.
        blank = True

    returnNow, value = _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg)
    if returnNow:
        return value

    # Validate against choices.
    if value in choices:
        return value
    if numbered and value.isdigit() and 0 < int(value) <= len(choices): # value must be 1 to len(choices)
        # Numbered options begins at 1, not 0.
        return choices[int(value) - 1] # -1 because the numbers are 1 to len(choices) but the index are 0 to len(choices) - 1
    if lettered and len(value) == 1 and value.isalpha() and 0 < ord(value.upper()) - 64 <= len(choices):
        # Lettered options are always case-insensitive.
        return choices[ord(value.upper()) - 65]
    if not caseSensitive and value.upper() in [choice.upper() for choice in choices]:
        # Return the original item in choices that value has a case-insensitive match with.
        return choices[[choice.upper() for choice in choices].index(value.upper())]

    _raiseValidationException(_('%r is not a valid choice.') % (_errstr(value)), excMsg)