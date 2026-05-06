def usernameAvailable(self, username, domain):
        """
        Check to see if a username is available for the user to select.
        """
        if len(username) < 2:
            return [False, u"Username too short"]
        for char in u"[ ,:;<>@()!\"'%&\\|\t\b":
            if char in username:
                return [False,
                        u"Username contains invalid character: '%s'" % char]

        # The localpart is acceptable if it can be parsed as the local part
        # of an RFC 2821 address.
        try:
            parseAddress("<%s@example.com>" % (username,))
        except ArgumentError:
            return [False, u"Username fails to parse"]

        # The domain is acceptable if it is one which we actually host.
        if domain not in self.getAvailableDomains():
            return [False, u"Domain not allowed"]

        query = self.store.query(userbase.LoginMethod,
                                 AND(userbase.LoginMethod.localpart == username,
                                     userbase.LoginMethod.domain == domain))
        return [not bool(query.count()), u"Username already taken"]