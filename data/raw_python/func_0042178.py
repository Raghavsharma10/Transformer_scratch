def relative_field(field, parent):
    """
    RETURN field PATH WITH RESPECT TO parent
    """
    if parent==".":
        return field

    field_path = split_field(field)
    parent_path = split_field(parent)
    common = 0
    for f, p in _builtin_zip(field_path, parent_path):
        if f != p:
            break
        common += 1

    if len(parent_path) == common:
        return join_field(field_path[common:])
    else:
        dots = "." * (len(parent_path) - common)
        return dots + "." + join_field(field_path[common:])