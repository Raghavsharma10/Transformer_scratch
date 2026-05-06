def parse_nhx(NHX_string):
    """ 
    NHX format:  [&&NHX:prop1=value1:prop2=value2] 
    MB format: ((a[&Z=1,Y=2], b[&Z=1,Y=2]):1.0[&L=1,W=0], ...
    """
    # store features
    ndict = {}

    # parse NHX or MB features
    if "[&&NHX:" in NHX_string:
        NHX_string = NHX_string.replace("[&&NHX:", "")
        NHX_string = NHX_string.replace("]", "")
        
        for field in NHX_string.split(":"):
            try:
                pname, pvalue = field.split("=")
                ndict[pname] = pvalue
            except ValueError as e:
                raise NewickError('Invalid NHX format %s' % field)
    return ndict