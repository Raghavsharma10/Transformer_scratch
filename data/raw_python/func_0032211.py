def _makeStoreOwnerPerson(self):
        """
        Make a L{Person} representing the owner of the store that this
        L{Organizer} is installed in.

        @rtype: L{Person}
        """
        if self.store is None:
            return None
        userInfo = self.store.findFirst(signup.UserInfo)
        name = u''
        if userInfo is not None:
            name = userInfo.realName
        account = self.store.findUnique(LoginAccount,
                                        LoginAccount.avatars == self.store, None)
        ownerPerson = self.createPerson(name)
        if account is not None:
            for method in (self.store.query(
                    LoginMethod,
                    attributes.AND(LoginMethod.account == account,
                                   LoginMethod.internal == False))):
                self.createContactItem(
                    EmailContactType(self.store),
                    ownerPerson, dict(
                        email=method.localpart + u'@' + method.domain))
        return ownerPerson