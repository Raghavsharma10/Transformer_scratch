def _is_custom_manager_attribute(node):
    """Checks if the attribute is a valid attribute for a queryset manager.
    """

    attrname = node.attrname
    if not name_is_from_qs(attrname):
        return False

    for attr in node.get_children():
        inferred = safe_infer(attr)
        funcdef = getattr(inferred, '_proxied', None)
        if _is_custom_qs_manager(funcdef):
            return True

    return False