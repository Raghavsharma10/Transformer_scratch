def unsuspendTabProviders(installation):
    """
    Remove suspension facades and replace them with their originals.
    """
    if not installation.suspended:
        raise RuntimeError("Installation not suspended")
    powerups = list(installation.allPowerups)
    allSNEs = list(powerups[0].store.powerupsFor(ISuspender))
    for p in powerups:
        for sne in allSNEs:
            if sne.originalNE is p:
                p.store.powerDown(sne, INavigableElement)
                p.store.powerDown(sne, ISuspender)
                p.store.powerUp(p, INavigableElement)
                sne.deleteFromStore()
    installation.suspended = False