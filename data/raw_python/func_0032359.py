def createUser(self, realName, username, domain, password, emailAddress):
        """
        Create a user, storing some associated metadata in the user's store,
        i.e. their first and last names (as a L{UserInfo} item), and a
        L{axiom.userbase.LoginMethod} allowing them to login with their email
        address.

        @param realName: the real name of the user.
        @type realName: C{unicode}

        @param username: the user's username.  they will be able to login with
            this.
        @type username: C{unicode}

        @param domain: the local domain - used internally to turn C{username}
            into a localpart@domain style string .
        @type domain: C{unicode}

        @param password: the password to be used for the user's account.
        @type password: C{unicode}

        @param emailAddress: the user's external email address.  they will be
            able to login with this also.
        @type emailAddress: C{unicode}

        @rtype: C{NoneType}
        """
        # XXX This method should be called in a transaction, it shouldn't
        # start a transaction itself.
        def _():
            loginsystem = self.store.findUnique(userbase.LoginSystem)

            # Create an account with the credentials they specified,
            # making it internal since it belongs to us.
            acct = loginsystem.addAccount(username, domain, password,
                                          verified=True, internal=True)

            # Create an external login method associated with the email
            # address they supplied, as well.  This creates an association
            # between that external address and their account object,
            # allowing password reset emails to be sent and letting them log
            # in to this account using that address as a username.
            emailPart, emailDomain = emailAddress.split("@")
            acct.addLoginMethod(emailPart, emailDomain, protocol=u"email",
                                verified=False, internal=False)
            substore = IBeneficiary(acct)
            # Record some of that signup information in case application
            # objects are interested in it.
            UserInfo(store=substore, realName=realName)
            self.product.installProductOn(substore)
        self.store.transact(_)