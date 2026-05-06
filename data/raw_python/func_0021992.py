def _check_authorization(cls, authzr, identifier):
        """
        Check that the authorization we got is the one we expected.
        """
        if authzr.body.identifier != identifier:
            raise errors.UnexpectedUpdate(authzr)
        return authzr