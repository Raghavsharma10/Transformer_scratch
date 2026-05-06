def is_module_function(obj, prop):
    """
    Checking and setting type to MODULE_FUNCTION
    Args:
        obj: ModuleType
        prop: FunctionType
    Return:
        Boolean
    Raise:
        prop_type_error: When the type of prop is not valid
        prop_in_obj_error: When prop is not in the obj(module/class)
        prop_is_func_error: When prop is not a callable stuff
    """
    python_version = sys.version_info[0]
    if python_version == 3:
        unicode = str

    if prop and (isinstance(prop, str) or isinstance(prop, unicode)): #property
        if prop in dir(obj):
            if (
                    isinstance(getattr(obj, prop), FunctionType)
                    or isinstance(getattr(obj, prop), BuiltinFunctionType)
                    or inspect.ismethod(getattr(obj, prop))
            ):
            #inspect.ismethod for python2.7
            #isinstance(...) for python3.x
                return True
            else:
                ErrorHandler.prop_is_func_error(obj, prop)
        else:
            ErrorHandler.prop_in_obj_error(obj, prop)
    elif prop:
        ErrorHandler.prop_type_error(prop)
    return False