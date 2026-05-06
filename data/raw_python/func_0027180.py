def impersonate(self, user, enterprise):
        """ Impersonate a user in a enterprise

            Args:
                user: the name of the user to impersonate
                enterprise: the name of the enterprise where to use impersonation
        """

        if not user or not enterprise:
            raise ValueError('You must set a user name and an enterprise name to begin impersonification')

        self._is_impersonating = True
        self._impersonation = "%s@%s" % (user, enterprise)