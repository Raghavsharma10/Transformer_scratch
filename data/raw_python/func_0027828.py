def insertUserStore(siteStore, userStorePath):
    """
    Move the SubStore at the indicated location into the given site store's
    directory and then hook it up to the site store's authentication database.

    @type siteStore: C{Store}
    @type userStorePath: C{FilePath}
    """
    # The following may, but does not need to be in a transaction, because it
    # is merely an attempt to guess a reasonable filesystem name to use for
    # this avatar.  The user store being operated on is expected to be used
    # exclusively by this process.
    ls = siteStore.findUnique(LoginSystem)
    unattachedSubStore = Store(userStorePath)
    for lm in unattachedSubStore.query(LoginMethod,
                                       LoginMethod.account == unattachedSubStore.findUnique(LoginAccount),
                                       sort=LoginMethod.internal.descending):
        if ls.accountByAddress(lm.localpart, lm.domain) is None:
            localpart, domain = lm.localpart, lm.domain
            break
    else:
        raise AllNamesConflict()

    unattachedSubStore.close()

    insertLocation = siteStore.newFilePath('account', domain, localpart + '.axiom')
    insertParentLoc = insertLocation.parent()
    if not insertParentLoc.exists():
        insertParentLoc.makedirs()
    if insertLocation.exists():
        raise DatabaseDirectoryConflict()
    userStorePath.moveTo(insertLocation)
    ss = SubStore(store=siteStore, storepath=insertLocation)
    attachedStore = ss.open()
    # migrateUp() manages its own transactions because it interacts with two
    # different stores.
    attachedStore.findUnique(LoginAccount).migrateUp()