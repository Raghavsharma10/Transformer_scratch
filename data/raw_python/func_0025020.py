def assert_has_permission(self, scope_required):
        """
        Warn that the required scope is not found in the scopes
        granted to the currently authenticated user.

        ::

            # The admin user should have client admin permissions
            uaa.assert_has_permission('admin', 'clients.admin')

        """
        if not self.authenticated:
            raise ValueError("Must first authenticate()")

        if scope_required not in self.get_scopes():
            logging.warning("Authenticated as %s" % (self.client['id']))
            logging.warning("Have scopes: %s" % (str.join(',', self.get_scopes())))
            logging.warning("Insufficient scope %s for operation" % (scope_required))

            raise ValueError("Client does not have permission.")

        return True