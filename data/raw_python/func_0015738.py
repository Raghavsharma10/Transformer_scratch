def StructureAttribute(struct_info):
    """Creates a new struct class."""

    # Copy the template and add the gtype
    cls_dict = dict(_Structure.__dict__)
    cls = type(struct_info.name, _Structure.__bases__, cls_dict)
    cls.__module__ = struct_info.namespace
    cls.__gtype__ = PGType(struct_info.g_type)
    cls._size = struct_info.size
    cls._is_gtype_struct = struct_info.is_gtype_struct

    # Add methods
    for method_info in struct_info.get_methods():
        add_method(method_info, cls)

    # Add fields
    for field_info in struct_info.get_fields():
        field_name = escape_identifier(field_info.name)
        attr = FieldAttribute(field_name, field_info)
        setattr(cls, field_name, attr)

    return cls