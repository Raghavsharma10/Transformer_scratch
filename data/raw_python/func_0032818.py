def _routerForAccount(self, identifier):
        """
        Locate an avatar by the username and domain portions of an
        L{Identifier}, so that we can deliver a message to the appropriate
        user.
        """
        acct = self.loginSystem.accountByAddress(identifier.localpart,
                                                 identifier.domain)
        return IMessageRouter(acct, None)