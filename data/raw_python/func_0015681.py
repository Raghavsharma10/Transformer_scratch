def add_method(info, target_cls, virtual=False, dont_replace=False):
    """Add a method to the target class"""

    # escape before prefixing, like pygobject
    name = escape_identifier(info.name)
    if virtual:
        name = "do_" + name
        attr = VirtualMethodAttribute(info, target_cls, name)
    else:
        attr = MethodAttribute(info, target_cls, name)

    if dont_replace and hasattr(target_cls, name):
        return

    setattr(target_cls, name, attr)