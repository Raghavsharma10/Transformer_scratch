def registerAttributeCopyingUpgrader(itemType, fromVersion, toVersion, postCopy=None):
    """
    Register an upgrader for C{itemType}, from C{fromVersion} to C{toVersion},
    which will copy all attributes from the legacy item to the new item.  If
    postCopy is provided, it will be called with the new item after upgrading.

    @param itemType: L{axiom.item.Item} subclass
    @param postCopy: a callable of one argument
    @return: None
    """
    def upgrader(old):
        newitem = old.upgradeVersion(itemType.typeName, fromVersion, toVersion,
                                     **dict((str(name), getattr(old, name))
                                            for (name, _) in old.getSchema()))
        if postCopy is not None:
            postCopy(newitem)
        return newitem
    registerUpgrader(upgrader, itemType.typeName, fromVersion, toVersion)