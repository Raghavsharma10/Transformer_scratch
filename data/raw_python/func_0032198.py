def _descriptiveIdentifier(contactType):
    """
    Get a descriptive identifier for C{contactType}, taking into account the
    fact that it might not have implemented the C{descriptiveIdentifier}
    method.

    @type contactType: L{IContactType} provider.

    @rtype: C{unicode}
    """
    descriptiveIdentifierMethod = getattr(
        contactType, 'descriptiveIdentifier', None)
    if descriptiveIdentifierMethod is not None:
        return descriptiveIdentifierMethod()
    warn(
        "IContactType now has the 'descriptiveIdentifier'"
        " method, %s did not implement it" % (contactType.__class__,),
        category=PendingDeprecationWarning)
    return _objectToName(contactType).decode('ascii')