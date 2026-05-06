def _parse_extra_features(node, NHX_string):
    """ 
    Reads node's extra data form its NHX string. NHX uses this
    format:  [&&NHX:prop1=value1:prop2=value2] 
    """
    NHX_string = NHX_string.replace("[&&NHX:", "")
    NHX_string = NHX_string.replace("]", "")
    for field in NHX_string.split(":"):
        try:
            pname, pvalue = field.split("=")
        except ValueError as e:
            raise NewickError('Invalid NHX format %s' %field)
        node.add_feature(pname, pvalue)