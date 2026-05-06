def toString(value, mode):
    """ Converts angle float to string. 
    Mode refers to LAT/LON.
    
    """
    string = angle.toString(value)
    sign = string[0]
    separator = CHAR[mode][sign]
    string = string.replace(':', separator, 1)
    return string[1:]