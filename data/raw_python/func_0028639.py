def replicate_methods(srcObj, dstObj):
    """Replicate callable methods from a `srcObj` to `dstObj` (generally a wrapper object). 
    
    @param srcObj: source object
    @param dstObj: destination object of the same type.
    @return : none
    
    Implementer notes: 
    1. Once the methods are mapped from the `srcObj` to the `dstObj`, the method calls will 
       not get "routed" through `__getattr__` method (if implemented) in `type(dstObj)` class.
    2. An example of what a 'key' and 'value' look like:
       key: MakeSequential
       value: <bound method IOpticalSystem.MakeSequential of 
              <win32com.gen_py.ZOSAPI_Interfaces.IOpticalSystem instance at 0x77183968>>
    """
    # prevent methods that we intend to specialize from being mapped. The specialized 
    # (overridden) methods are methods with the same name as the corresponding method in 
    # the source ZOS API COM object written for each ZOS API COM object in an associated 
    # python script such as i_analyses_methods.py for I_Analyses
    overridden_methods = get_callable_method_dict(type(dstObj)).keys()
    #overridden_attrs = [each for each in type(dstObj).__dict__.keys() if not each.startswith('_')]
    # 

    def zos_wrapper_deco(func):
        def wrapper(*args, **kwargs):
            return wrapped_zos_object(func(*args, **kwargs))
        varnames = func.im_func.func_code.co_varnames # alternative is to use inspect.getargspec
        params = [par for par in varnames if par not in ('self', 'ret')] # removes 'self' and 'ret'
        wrapper.__doc__ = func.im_func.func_name + '(' + ', '.join(params) + ')' 
        return wrapper 
    #
    for key, value in get_callable_method_dict(srcObj).items():
        if key not in overridden_methods:
            setattr(dstObj, key, zos_wrapper_deco(value))