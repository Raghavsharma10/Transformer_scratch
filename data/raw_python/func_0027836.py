def addLoginMethod(self, localpart, domain, protocol=ANY_PROTOCOL, verified=False, internal=False):
        """
        Add a login method to this account, propogating up or down as necessary
        to site store or user store to maintain consistency.
        """
        # Out takes you west or something
        if self.store.parent is None:
            # West takes you in
            otherStore = self.avatars.open()
            peer = otherStore.findUnique(LoginAccount)
        else:
            # In takes you east
            otherStore = self.store.parent
            subStoreItem = self.store.parent.getItemByID(self.store.idInParent)
            peer = otherStore.findUnique(LoginAccount,
                                         LoginAccount.avatars == subStoreItem)

        # Up and down take you home
        for store, account in [(otherStore, peer), (self.store, self)]:
            store.findOrCreate(LoginMethod,
                               account=account,
                               localpart=localpart,
                               domain=domain,
                               protocol=protocol,
                               verified=verified,
                               internal=internal)