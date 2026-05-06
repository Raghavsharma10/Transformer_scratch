def createUser(self, localpart, domain, password=None):
        """
        Create a new, blank user account with the given name and domain and, if
        specified, with the given password.

        @type localpart: C{unicode}
        @param localpart: The local portion of the username.  ie, the
        C{'alice'} in C{'alice@example.com'}.

        @type domain: C{unicode}
        @param domain: The domain portion of the username.  ie, the
        C{'example.com'} in C{'alice@example.com'}.

        @type password: C{unicode} or C{None}
        @param password: The password to associate with the new account.  If
        C{None}, generate a new password automatically.
        """
        loginSystem = self.browser.store.parent.findUnique(userbase.LoginSystem)
        if password is None:
            password = u''.join([random.choice(string.ascii_letters + string.digits) for i in xrange(8)])
        loginSystem.addAccount(localpart, domain, password)