def ObjectAttribute(obj_info):
    """Creates a GObject class.

    It inherits from the base class and all interfaces it implements.
    """

    if obj_info.name == "Object" and obj_info.namespace == "GObject":
        cls = Object
    else:
        # Get the parent class
        parent_obj = obj_info.get_parent()
        if parent_obj:
            attr = import_attribute(parent_obj.namespace, parent_obj.name)
            bases = (attr,)
        else:
            bases = (object,)

        # Get all object interfaces
        ifaces = []
        for interface in obj_info.get_interfaces():
            attr = import_attribute(interface.namespace, interface.name)
            # only add interfaces if the base classes don't have it
            for base in bases:
                if attr in base.__mro__:
                    break
            else:
                ifaces.append(attr)

        # Combine them to a base class list
        if ifaces:
            bases = tuple(list(bases) + ifaces)

        # Create a new class
        cls = type(obj_info.name, bases, dict())

    cls.__module__ = obj_info.namespace

    # Set root to unowned= False and InitiallyUnowned=True
    if obj_info.namespace == "GObject":
        if obj_info.name == "InitiallyUnowned":
            cls._unowned = True
        elif obj_info.name == "Object":
            cls._unowned = False

    # GType
    cls.__gtype__ = PGType(obj_info.g_type)

    if not obj_info.fundamental:
        # Constructor cache
        cls._constructors = {}

        # Properties
        setattr(cls, PROPS_NAME, PropertyAttribute(obj_info))

        # Signals
        cls.signals = SignalsAttribute(obj_info)

        # Signals
        cls.__sigs__ = {}
        for sig_info in obj_info.get_signals():
            signal_name = sig_info.name
            cls.__sigs__[signal_name] = sig_info

    # Add constants
    for constant in obj_info.get_constants():
        constant_name = constant.name
        attr = ConstantAttribute(constant)
        setattr(cls, constant_name, attr)

    # Fields
    for field in obj_info.get_fields():
        field_name = escape_identifier(field.name)
        attr = FieldAttribute(field_name, field)
        setattr(cls, field_name, attr)

    # Add methods
    for method_info in obj_info.get_methods():
        # we implement most of the base object ourself
        add_method(method_info, cls, dont_replace=cls is Object)

    # VFuncs
    for vfunc_info in obj_info.get_vfuncs():
        add_method(vfunc_info, cls, virtual=True)

    cs_info = obj_info.get_class_struct()
    if cs_info:
        class_struct = import_attribute(cs_info.namespace, cs_info.name)
    else:
        class_struct = None

    # XXX ^ 2
    def get_class_struct(cls, type_=None):
        """Returns the class struct casted to the passed type"""

        if type_ is None:
            type_ = class_struct

        if type_ is None:
            return None

        ptr = cls.__gtype__._type.class_ref()
        return type_._from_pointer(ptr)

    setattr(cls, "_get_class_struct", classmethod(get_class_struct))

    return cls