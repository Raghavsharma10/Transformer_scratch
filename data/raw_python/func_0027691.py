def addAccount(self, siteStore, username, domain, password):
        """
        Create a new account in the given store.

        @param siteStore: A site Store to which login credentials will be
        added.
        @param username: Local part of the username for the credentials to add.
        @param domain: Domain part of the username for the credentials to add.
        @param password: Password for the credentials to add.
        @rtype: L{LoginAccount}
        @return: The added account.
        """
        for ls in siteStore.query(userbase.LoginSystem):
            break
        else:
            ls = self.installOn(siteStore)
        try:
            acc = ls.addAccount(username, domain, password)
        except userbase.DuplicateUser:
            raise usage.UsageError("An account by that name already exists.")
        return acc