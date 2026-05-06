def registerDeletionUpgrader(itemType, fromVersion, toVersion):
    """
    Register an upgrader for C{itemType}, from C{fromVersion} to C{toVersion},
    which will delete the item from the database.

    @param itemType: L{axiom.item.Item} subclass
    @return: None
    """
    # XXX This should actually do something more special so that a new table is
    # not created and such.
    def upgrader(old):
        old.deleteFromStore()
        return None
    registerUpgrader(upgrader, itemType.typeName, fromVersion, toVersion)