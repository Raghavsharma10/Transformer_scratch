def passwordReset1to2(old):
    """
    Power down and delete the item
    """
    new = old.upgradeVersion(old.typeName, 1, 2, installedOn=None)
    for iface in new.store.interfacesFor(new):
        new.store.powerDown(new, iface)
    new.deleteFromStore()