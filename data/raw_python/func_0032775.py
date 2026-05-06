def _checkConflictingNames(interfaces):
    """
    Raise an exception if any of the names present in the given interfaces
    conflict with each other.

    @param interfaces: a list of Zope Interface objects.

    @return: None

    @raise ConflictingNames: if any of the attributes of the provided
    interfaces are the same, and they do not have a common base interface which
    provides that name.
    """
    names = {}
    for interface in interfaces:
        for name in interface:
            if name in names:
                otherInterface = names[name]
                parent = _commonParent(interface, otherInterface)
                if parent is None or name not in parent:
                    raise ConflictingNames("%s conflicts with %s over %s" % (
                            interface, otherInterface, name))
            names[name] = interface