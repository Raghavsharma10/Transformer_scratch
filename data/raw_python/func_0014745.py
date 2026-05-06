def convertToPositiveInt(val=None, invalidDefault=0):
    '''
        convertToPositiveInt - Convert to a positive integer, and if invalid use a given value
    '''
    if val is None:
        return invalidDefault

    try:
        val = int(val)
    except:
        return invalidDefault

    if val < 0:
        return invalidDefault

    return val