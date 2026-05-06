def authenticate(self, account_name, password):
        """
        Authenticates zimbra account.
        @param account_name: account email address
        @param password: account password
        @raise AuthException: if authentication fails
        @raise SoapException: if soap communication fails
        """
        self.auth_token = self.authenticator.authenticate(self.transport,
                                                          account_name,
                                                          password)