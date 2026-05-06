def is_field_method(node):
    """Checks if a call to a field instance method is valid. A call is
    valid if the call is a method of the underlying type. So, in a StringField
    the methods from str are valid, in a ListField the methods from list are
    valid and so on..."""
    name = node.attrname
    parent = node.last_child()
    inferred = safe_infer(parent)

    if not inferred:
        return False

    for cls_name, inst in FIELD_TYPES.items():
        if node_is_instance(inferred, cls_name) and hasattr(inst, name):
            return True

    return False