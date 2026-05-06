def suspendJustTabProviders(installation):
    """
    Replace INavigableElements with facades that indicate their suspension.
    """
    if installation.suspended:
        raise RuntimeError("Installation already suspended")
    powerups = list(installation.allPowerups)
    for p in powerups:
        if INavigableElement.providedBy(p):
            p.store.powerDown(p, INavigableElement)
            sne = SuspendedNavigableElement(store=p.store, originalNE=p)
            p.store.powerUp(sne, INavigableElement)
            p.store.powerUp(sne, ISuspender)
    installation.suspended = True