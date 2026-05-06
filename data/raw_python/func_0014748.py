def convertToIntRange(val, minValue, maxValue, invalidDefault, emptyValue=''):
    '''
        converToIntRange - Convert input value to an integer within a certain range
            
            @param val <None/str/int/float> - The input value

            @param minValue <None/int> - The minimum value (inclusive), or None if no minimum

            @param maxValue <None/int> - The maximum value (inclusive), or None if no maximum

            @param invalidDefault <None/str/Exception> - The value to return if "val" is not empty string/None
                                                           and "val" is not in #possibleValues

                     If instantiated Exception (like ValueError('blah')):  Raise this exception

                     If an Exception type ( like ValueError ) - Instantiate and raise this exception type

                     Otherwise, use this raw value

            @param emptyValue Default '', used for an empty value (empty string or None)
                

    '''
    from .utils import tostr

    # If null, retain null
    if val is None or val == '':
        if emptyValue is EMPTY_IS_INVALID:
            return _handleInvalid(invalidDefault)
        return emptyValue

    try:
        val = int(val)
    except ValueError:
        return _handleInvalid(invalidDefault)

    if minValue is not None and val < minValue:
        return _handleInvalid(invalidDefault)
    if maxValue is not None and val > maxValue:
        return _handleInvalid(invalidDefault)

    return val