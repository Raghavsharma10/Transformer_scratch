def _prevalidationCheck(value, blank, strip, allowlistRegexes, blocklistRegexes, excMsg=None):
    """Returns a tuple of two values: the first is a bool that tells the caller
    if they should immediately return True, the second is a new, possibly stripped
    value to replace the value passed for value parameter.

    We'd want the caller immediately return value in some cases where further
    validation isn't needed, such as if value is blank and blanks are
    allowed, or if value matches an allowlist or blocklist regex.

    This function is called by the validate*() functions to perform some common
    housekeeping."""

    # TODO - add a allowlistFirst and blocklistFirst to determine which is checked first. (Right now it's allowlist)

    value = str(value)

    # Optionally strip whitespace or other characters from value.
    value = _getStrippedValue(value, strip)

    # Validate for blank values.
    if not blank and value == '':
        # value is blank but blanks aren't allowed.
        _raiseValidationException(_('Blank values are not allowed.'), excMsg)
    elif blank and value == '':
        return True, value # The value is blank and blanks are allowed, so return True to indicate that the caller should return value immediately.

    # NOTE: We check if something matches the allow-list first, then we check the block-list second.

    # Check the allowlistRegexes.
    if allowlistRegexes is not None:
        for regex in allowlistRegexes:
            if isinstance(regex, re.Pattern):
                if regex.search(value, re.IGNORECASE) is not None:
                    return True, value # The value is in the allowlist, so return True to indicate that the caller should return value immediately.
            else:
                if re.search(regex, value, re.IGNORECASE) is not None:
                    return True, value # The value is in the allowlist, so return True to indicate that the caller should return value immediately.

    # Check the blocklistRegexes.
    if blocklistRegexes is not None:
        for blocklistRegexItem in blocklistRegexes:
            if isinstance(blocklistRegexItem, str):
                regex, response = blocklistRegexItem, DEFAULT_BLOCKLIST_RESPONSE
            else:
                regex, response = blocklistRegexItem

            if isinstance(regex, re.Pattern) and regex.search(value, re.IGNORECASE) is not None:
                _raiseValidationException(response, excMsg) # value is on a blocklist
            elif re.search(regex, value, re.IGNORECASE) is not None:
                _raiseValidationException(response, excMsg) # value is on a blocklist

    return False, value