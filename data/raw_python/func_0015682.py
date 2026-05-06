def InterfaceAttribute(iface_info):
    """Creates a GInterface class"""

    # Create a new class
    cls = type(iface_info.name, (InterfaceBase,), dict(_Interface.__dict__))
    cls.__module__ = iface_info.namespace

    # GType
    cls.__gtype__ = PGType(iface_info.g_type)

    # Properties
    cls.props = PropertyAttribute(iface_info)

    # Signals
    cls.signals = SignalsAttribute(iface_info)

    # Add constants
    for constant in iface_info.get_constants():
        constant_name = constant.name
        attr = ConstantAttribute(constant)
        setattr(cls, constant_name, attr)

    # Add methods
    for method_info in iface_info.get_methods():
        add_method(method_info, cls)

    # VFuncs
    for vfunc_info in iface_info.get_vfuncs():
        add_method(vfunc_info, cls, virtual=True)

    cls._sigs = {}

    is_info = iface_info.get_iface_struct()
    if is_info:
        iface_struct = import_attribute(is_info.namespace, is_info.name)
    else:
        iface_struct = None

    def get_iface_struct(cls):
        if not iface_struct:
            return None

        ptr = cls.__gtype__._type.default_interface_ref()
        if not ptr:
            return None
        return iface_struct._from_pointer(addressof(ptr.contents))

    setattr(cls, "_get_iface_struct", classmethod(get_iface_struct))

    return cls