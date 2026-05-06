def get_referencer(registry):
    """
    Get the referencer class

    :rtype: pyramid_urireferencer.referencer.AbstractReferencer
    """
    # Argument might be a config or request
    regis = getattr(registry, 'registry', None)
    if regis is None:
        regis = registry
    return regis.queryUtility(IReferencer)