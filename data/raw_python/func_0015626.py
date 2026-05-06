def get_field_type(info):
    """A field python type"""

    type_ = info.get_type()

    cls = get_field_class(type_)

    field = cls(info, type_, None)
    field.setup()
    return field.py_type