def cloneInto(self, newStore, avatars):
        """
        Create a copy of this LoginAccount and all associated LoginMethods in a different Store.

        Return the copied LoginAccount.
        """
        la = LoginAccount(store=newStore,
                          password=self.password,
                          avatars=avatars,
                          disabled=self.disabled)
        for siteMethod in self.store.query(LoginMethod,
                                           LoginMethod.account == self):
            LoginMethod(store=newStore,
                        localpart=siteMethod.localpart,
                        domain=siteMethod.domain,
                        internal=siteMethod.internal,
                        protocol=siteMethod.protocol,
                        verified=siteMethod.verified,
                        account=la)
        return la