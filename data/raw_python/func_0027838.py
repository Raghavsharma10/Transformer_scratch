def addAccount(self, username, domain, password, avatars=None,
                   protocol=u'email', disabled=0, internal=False,
                   verified=True):
        """
        Create a user account, add it to this LoginBase, and return it.

        This method must be called within a transaction in my store.

        @param username: the user's name.

        @param domain: the domain part of the user's name [XXX TODO: this
        really ought to say something about whether it's a Q2Q domain, a SIP
        domain, an HTTP realm, or an email address domain - right now the
        assumption is generally that it's an email address domain, but not
        always]

        @param password: A shared secret.

        @param avatars: (Optional).  A SubStore which, if passed, will be used
        by cred as the target of all adaptations for this user.  By default, I
        will create a SubStore, and plugins can be installed on that substore
        using the powerUp method to provide implementations of cred client
        interfaces.

        @raise DuplicateUniqueItem: if the 'avatars' argument already contains
        a LoginAccount.

        @return: an instance of a LoginAccount, with all attributes filled out
        as they are passed in, stored in my store.
        """

        # unicode(None) == u'None', kids.
        if username is not None:
            username = unicode(username)
        if domain is not None:
            domain = unicode(domain)
        if password is not None:
            password = unicode(password)

        if self.accountByAddress(username, domain) is not None:
            raise DuplicateUser(username, domain)
        if avatars is None:
            avatars = self.makeAvatars(domain, username)

        subStore = avatars.open()

        # create this unconditionally; as the docstring says, we must be run
        # within a transaction, so if something goes wrong in the substore
        # transaction this item's creation will be reverted...
        la = LoginAccount(store=self.store,
                          password=password,
                          avatars=avatars,
                          disabled=disabled)

        def createSubStoreAccountObjects():

            LoginAccount(store=subStore,
                         password=password,
                         disabled=disabled,
                         avatars=subStore)

            la.addLoginMethod(localpart=username,
                              domain=domain,
                              protocol=protocol,
                              internal=internal,
                              verified=verified)

        subStore.transact(createSubStoreAccountObjects)
        return la