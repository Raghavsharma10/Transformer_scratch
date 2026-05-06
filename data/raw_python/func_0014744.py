def convertBooleanStringToBoolean(val=None):
    '''
        convertBooleanStringToBoolean - Convert from a boolean attribute (string "true" / "false" ) into a booelan
    '''
    if not val:
        return False

    if hasattr(val, 'lower'):
        val = val.lower()

    if val == "false":
        return False
    return True