def parse_value(val):
    """Parse input string and return int, float or str depending on format.
    
    @param val: Input string.
    @return:    Value of type int, float or str.
        
    """
    
    mobj = re.match('(-{0,1}\d+)\s*(\sseconds|/\s*\w+)$',  val)
    if mobj:
        return int(mobj.group(1))
    mobj = re.match('(-{0,1}\d*\.\d+)\s*(\sseconds|/\s*\w+)$',  val)
    if mobj:
        return float(mobj.group(1))
    re.match('(-{0,1}\d+)\s*([GMK])B$',  val)
    if mobj:
        return int(mobj.group(1)) * memMultiplier[mobj.group(2)]
    mobj = re.match('(-{0,1}\d+(\.\d+){0,1})\s*\%$',  val)
    if mobj:
        return float(mobj.group(1)) / 100 
    return val