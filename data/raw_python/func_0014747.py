def convertPossibleValues(val, possibleValues, invalidDefault, emptyValue=''):
    '''
        convertPossibleValues - Convert input value to one of several possible values,
            
                                    with a default for invalid entries

            @param val <None/str> - The input value

            @param possibleValues list<str> - A list of possible values

            @param invalidDefault <None/str/Exception> - The value to return if "val" is not empty string/None
                                                           and "val" is not in #possibleValues

                     If instantiated Exception (like ValueError('blah')):  Raise this exception

                     If an Exception type ( like ValueError ) - Instantiate and raise this exception type

                     Otherwise, use this raw value

            @param emptyValue Default '', used for an empty value (empty string or None)
                

    '''
    from .utils import tostr

    # If null, retain null
    if val is None:
        if emptyValue is EMPTY_IS_INVALID:
            return _handleInvalid(invalidDefault)
        return emptyValue

    # Convert to a string
    val = tostr(val).lower()

    # If empty string, same as null
    if val == '':
        if emptyValue is EMPTY_IS_INVALID:
            return _handleInvalid(invalidDefault)
        return emptyValue

    # Check if this is a valid value
    if val not in possibleValues:
        return _handleInvalid(invalidDefault)

    return val