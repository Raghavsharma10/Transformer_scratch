def toFloat(value):
    """ Converts angle representation to float. 
    Accepts angles and strings such as "12W30:00".
    
    """
    if isinstance(value, str):
        # Find lat/lon char in string and insert angle sign
        value = value.upper()
        for char in ['N', 'S', 'E', 'W']:
            if char in value:
                value = SIGN[char] + value.replace(char, ':')
                break
    return angle.toFloat(value)