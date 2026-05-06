def _CollectHistory_(lookupType, fromVal, toVal, using={}, pattern=''):
    """
    Return a dictionary detailing what, if any, change was made to a record field

    :param string lookupType: what cleaning rule made the change; one of: genericLookup, genericRegex, fieldSpecificLookup, fieldSpecificRegex, normLookup, normRegex, normIncludes, deriveValue, copyValue, deriveRegex
    :param string fromVal: previous field value
    :param string toVal: new string value
    :param dict using: field values used to derive new values; only applicable for deriveValue, copyValue, deriveRegex
    :param string pattern: which regex pattern was matched to make the change; only applicable for genericRegex, fieldSpecificRegex, deriveRegex
    """

    histObj = {}

    if fromVal != toVal:
        histObj[lookupType] = {"from": fromVal, "to": toVal}

        if lookupType in ['deriveValue', 'deriveRegex', 'copyValue', 'normIncludes', 'deriveIncludes'] and using!='':
            histObj[lookupType]["using"] = using
        if lookupType in ['genericRegex', 'fieldSpecificRegex', 'normRegex', 'deriveRegex'] and pattern!='':
            histObj[lookupType]["pattern"] = pattern

    return histObj