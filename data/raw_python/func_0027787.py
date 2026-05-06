def upgradeParentHook3to4(old):
    """
    Copy C{loginAccount} to C{subStore} and remove the installation marker.
    """
    new = old.upgradeVersion(
        old.typeName, 3, 4, subStore=old.loginAccount)
    uninstallFrom(new, new.store)
    return new