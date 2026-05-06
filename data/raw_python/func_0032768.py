def _interfacesToNames(interfaces):
    """
    Convert from a list of interfaces to a unicode string of names suitable for
    storage in the database.

    @param interfaces: an iterable of Interface objects.

    @return: a unicode string, a comma-separated list of names of interfaces.

    @raise ConflictingNames: if any of the names conflict: see
    L{_checkConflictingNames}.
    """
    if interfaces is ALL_IMPLEMENTED:
        names = ALL_IMPLEMENTED_DB
    else:
        _checkConflictingNames(interfaces)
        names = u','.join(map(qual, interfaces))
    return names