def _handleInvalid(invalidDefault):
    '''
        _handleInvalid - Common code for raising / returning an invalid value

            @param invalidDefault <None/str/Exception> - The value to return if "val" is not empty string/None
                                                           and "val" is not in #possibleValues

                     If instantiated Exception (like ValueError('blah')):  Raise this exception

                     If an Exception type ( like ValueError ) - Instantiate and raise this exception type

                     Otherwise, use this raw value
    '''
    # If not
    #   If an instantiated Exception, raise that exception
    try:
        isInstantiatedException = bool( issubclass(invalidDefault.__class__, Exception) )
    except:
        isInstantiatedException = False

    if isInstantiatedException:
        raise invalidDefault
    else:
        try:
            isExceptionType = bool( issubclass( invalidDefault, Exception) )
        except TypeError:
            isExceptionType = False

        #   If an Exception type, instantiate and raise
        if isExceptionType:
            raise invalidDefault()
        else:
        #   Otherwise, just return invalidDefault itself
            return invalidDefault