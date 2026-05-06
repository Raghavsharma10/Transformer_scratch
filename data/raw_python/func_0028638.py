def get_callable_method_dict(obj):
    """Returns a dictionary of callable methods of object `obj`.

    @param obj: ZOS API Python COM object
    @return: a dictionary of callable methods
    
    Notes: 
    the function only returns the callable attributes that are listed by dir() 
    function. Properties are not returned.
    """
    methodDict = {}
    for methodStr in dir(obj):
        method = getattr(obj, methodStr, 'none')
        if callable(method) and not methodStr.startswith('_'):
            methodDict[methodStr] = method
    return methodDict