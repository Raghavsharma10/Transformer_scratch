def managed_wrapper_class_factory(zos_obj):
    """Creates and returns a wrapper class of a ZOS object, exposing the ZOS objects 
    methods and propertis, and patching custom specialized attributes

    @param zos_obj: ZOS API Python COM object
    """
    cls_name = repr(zos_obj).split()[0].split('.')[-1]  
    dispatch_attr = '_' + cls_name.lower()  # protocol to be followed to store the ZOS COM object
    
    cdict = {}  # class dictionary

    # patch the properties of the base objects 
    base_cls_list = inheritance_dict.get(cls_name, None)
    if base_cls_list:
        for base_cls_name in base_cls_list:
            getters, setters = get_properties(_CastTo(zos_obj, base_cls_name))
            for each in getters:
                exec("p{} = ZOSPropMapper('{}', '{}', cast_to='{}')".format(each, dispatch_attr, each, base_cls_name), globals(), cdict)
            for each in setters:
                exec("p{} = ZOSPropMapper('{}', '{}', setter=True, cast_to='{}')".format(each, dispatch_attr, each, base_cls_name), globals(), cdict)

    # patch the property attributes of the given ZOS object
    getters, setters = get_properties(zos_obj)
    for each in getters:
        exec("p{} = ZOSPropMapper('{}', '{}')".format(each, dispatch_attr, each), globals(), cdict)
    for each in setters:
        exec("p{} = ZOSPropMapper('{}', '{}', setter=True)".format(each, dispatch_attr, each), globals(), cdict)
    
    def __init__(self, zos_obj):
        
        # dispatcher attribute
        cls_name = repr(zos_obj).split()[0].split('.')[-1] 
        dispatch_attr = '_' + cls_name.lower()    # protocol to be followed to store the ZOS COM object
        self.__dict__[dispatch_attr] = zos_obj
        self._dispatch_attr_value = dispatch_attr # used in __getattr__
        
        # Store base class object 
        self._base_cls_list = inheritance_dict.get(cls_name, None)

        # patch the methods of the base class(s) of the given ZOS object
        if self._base_cls_list:
            for base_cls_name in self._base_cls_list:
                replicate_methods(_CastTo(zos_obj, base_cls_name), self)

        # patch the methods of given ZOS object 
        replicate_methods(zos_obj, self)

        # mark object as wrapped to prevent it from being wrapped subsequently
        self._wrapped = True
    
    # Provide a way to make property calls without the prefix p
    def __getattr__(self, attrname):
        return wrapped_zos_object(getattr(self.__dict__[self._dispatch_attr_value], attrname))

    def __repr__(self):
        if type(self).__name__ == 'IZOSAPI_Application':
            repr_str = "{.__name__}(NumberOfOpticalSystems = {})".format(type(self), self.pNumberOfOpticalSystems)
        else:
            repr_str = "{.__name__}".format(type(self))
        return repr_str
        
    cdict['__init__'] = __init__
    cdict['__getattr__'] = __getattr__
    cdict['__repr__'] = __repr__
    
    # patch custom methods from python files imported as modules
    module_import_str = """
try: 
    from pyzos.zos_obj_override.{module:} import *
except ImportError:
    pass
""".format(module=cls_name.lower() + '_methods')
    exec(module_import_str, globals(), cdict)

    _ = cdict.pop('print_function', None)
    _ = cdict.pop('division', None)
    
    return type(cls_name, (), cdict)