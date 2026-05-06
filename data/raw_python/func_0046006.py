def authenticate(self, transport, account_name, password):
        """
        Authenticates account, if no password given tries to pre-authenticate.
        @param transport: transport to use for method calls
        @param account_name: account name
        @param password: account password
        @return: AuthToken if authentication succeeded
        @raise AuthException: if authentication fails
        """
        if not isinstance(transport, ZimbraClientTransport):
            raise ZimbraClientException('Invalid transport')

        if util.empty(account_name):
            raise AuthException('Empty account name')