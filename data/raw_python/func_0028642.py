def wrapped_zos_object(zos_obj):
    """Helper function to wrap ZOS API COM objects. 

    @param zos_obj : ZOS API Python COM object
    @return: instance of the wrapped ZOS API class. If the input object is not a ZOS-API
             COM object or if it is already wrapped, then the object is returned without
             wrapping.

    Notes:
    The function dynamically creates a wrapped class with all the provided methods, 
    properties, and custom methods monkey patched; and returns an instance of it.
    """
    if hasattr(zos_obj, '_wrapped') or ('CLSID' not in dir(zos_obj)):
        return zos_obj
    else:
        Class = managed_wrapper_class_factory(zos_obj)   
        return Class(zos_obj)