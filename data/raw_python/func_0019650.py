def parse_value(val, parsebool=False):
    """Parse input string and return int, float or str depending on format.
    
    @param val:       Input string.
    @param parsebool: If True parse yes / no, on / off as boolean.
    @return:          Value of type int, float or str.
        
    """
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except:
        pass
    if parsebool:
        if re.match('yes|on', str(val), re.IGNORECASE):
            return True
        elif re.match('no|off', str(val), re.IGNORECASE):
            return False
    return val