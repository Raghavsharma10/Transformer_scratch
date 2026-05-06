def convertToBooleanString(val=None):
    '''
        convertToBooleanString - Converts a value to either a string of "true" or "false"

            @param val <int/str/bool> - Value
    '''
    if hasattr(val, 'lower'):
        val = val.lower()

        # Technically, if you set one of these attributes (like "spellcheck") to a string of 'false',
        #   it gets set to true. But we will retain "false" here.
        if val in ('false', '0'):
            return 'false'
        else:
            return 'true'
    
    try:
        if bool(val):
            return "true"
    except:
        pass

    return "false"