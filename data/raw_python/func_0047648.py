def authenticate(self, transport, account_name, password=None):
        """
        Authenticates account using soap method.
        """
        Authenticator.authenticate(self, transport, account_name, password)

        if password == None:
            return self.pre_auth(transport, account_name)
        else:
            return self.auth(transport, account_name, password)