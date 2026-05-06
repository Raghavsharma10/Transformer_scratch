def listVersionHistory(store):
    """
    List the software package version history of store.
    """
    q = store.query(SystemVersion, sort=SystemVersion.creation.descending)
    return [sv.longWindedRepr() for sv in q]