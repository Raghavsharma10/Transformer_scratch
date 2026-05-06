def _add_referencer(registry):
    """
    Gets the Referencer from config and adds it to the registry.
    """
    referencer = registry.queryUtility(IReferencer)
    if referencer is not None:
        return referencer
    ref = registry.settings['urireferencer.referencer']
    url = registry.settings['urireferencer.registry_url']
    r = DottedNameResolver()
    registry.registerUtility(r.resolve(ref)(url), IReferencer)
    return registry.queryUtility(IReferencer)