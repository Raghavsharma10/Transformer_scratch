def convert_the_getters(getters):
    """
    A function used to prepare the arguments of calculator and atoms getter methods
    """
    return_list = []
    for getter in getters:
        
        if isinstance(getter,basestring):
            out_args = ""
            method_name = getter
            
        else:
            method_name, a = getter
            
            out_args = convert_the_args(a)
            
        return_list.append( (method_name, out_args) )
    return return_list