def randomEarlyShared(store, role):
    """
    If there are no explicitly-published public index pages to display, find a
    shared item to present to the user as first.
    """
    for r in role.allRoles():
        share = store.findFirst(Share, Share.sharedTo == r,
                                sort=Share.storeID.ascending)
        if share is not None:
            return share.sharedItem
    raise NoSuchShare("Why, that user hasn't shared anything at all!")