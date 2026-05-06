def permissions(self):
        """
        A list of strings describing permissions.

        See Facebook's exhaustive `Permissions Reference <http://developers.facebook.com/docs/authentication/permissions/>`_
        for a list of available permissions.
        """
        response = self.graph.get('%s/permissions' % self.id)

        permissions = []
        for permission, state in response['data'][0].items():
            permissions.append(permission)
        
        return permissions