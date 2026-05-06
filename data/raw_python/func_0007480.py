def add_grant(self, grant):
        """
        Adds a Grant that the provider should support.

        :param grant: An instance of a class that extends
                      :class:`oauth2.grant.GrantHandlerFactory`
        :type grant: oauth2.grant.GrantHandlerFactory
        """
        if hasattr(grant, "expires_in"):
            self.token_generator.expires_in[grant.grant_type] = grant.expires_in

        if hasattr(grant, "refresh_expires_in"):
            self.token_generator.refresh_expires_in = grant.refresh_expires_in

        self.grant_types.append(grant)