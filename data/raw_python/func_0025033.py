def get_user(self, id):
        """
        Returns details about the user for the given id.

        Use get_user_by_email() or get_user_by_username() for help
        identifiying the id.
        """
        self.assert_has_permission('scim.read')
        return self._get(self.uri + '/Users/%s' % (id))