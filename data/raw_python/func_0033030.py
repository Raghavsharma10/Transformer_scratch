def isAppStore(s):
    """
    Return whether the given store is an application store or not.
    @param s: A Store.
    """
    if s.parent is None:
        return False
    substore = s.parent.getItemByID(s.idInParent)
    return s.parent.query(InstalledOffering,
                          InstalledOffering.application == substore
                          ).count() > 0