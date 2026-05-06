def getrange(bch, fieldname):
    """get the ranges for this field"""
    keys = ['maximum', 'minimum', 'maximum<', 'minimum>', 'type']
    index = bch.objls.index(fieldname)
    fielddct_orig = bch.objidd[index]
    fielddct = copy.deepcopy(fielddct_orig)
    therange = {}
    for key in keys:
        therange[key] = fielddct.setdefault(key, None)
    if therange['type']:
        therange['type'] = therange['type'][0]
    if therange['type'] == 'real':
        for key in keys[:-1]:
            if therange[key]:
                therange[key] = float(therange[key][0])
    if therange['type'] == 'integer':
        for key in keys[:-1]:
            if therange[key]:
                therange[key] = int(therange[key][0])
    return therange