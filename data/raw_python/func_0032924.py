def _storeFromUsername(store, username):
    """
    Find the user store of the user with username C{store}

    @param store: site-store
    @type store: L{axiom.store.Store}

    @param username: the name a user signed up with
    @type username: C{unicode}

    @rtype: L{axiom.store.Store} or C{None}
    """
    lm = store.findUnique(
            userbase.LoginMethod,
            attributes.AND(
                userbase.LoginMethod.localpart == username,
                userbase.LoginMethod.internal == True),
            default=None)
    if lm is not None:
        return lm.account.avatars.open()