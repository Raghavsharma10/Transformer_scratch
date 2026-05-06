def convert_the_args(raw_args):
    """
    Function used to convert the arguments of methods
    """
    if not raw_args:
        return ""
    if isinstance(raw_args,dict):
        out_args = ", ".join([ "{}={}".format(k,v) for k,v in raw_args.iteritems() ])
        
    elif isinstance(raw_args,(list,tuple)):
        new_list = []
        for x in raw_args:
            if isinstance(x,basestring):
                new_list.append(x)
            elif isinstance(x,dict):
                new_list.append( ", ".join([ "{}={}".format(k,v) for k,v in x.iteritems() ]) )
            else:
                raise ValueError("Error preparing the getters")
        out_args = ", ".join(new_list)
    else:
        raise ValueError("Couldn't recognize list of getters")
    return out_args