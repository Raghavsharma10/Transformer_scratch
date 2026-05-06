def compare(self, previous_scopes):
        """
        Compares the scopes read from request with previously issued scopes.

        :param previous_scopes: A list of scopes.
        :return: ``True``
        """
        for scope in self.scopes:
            if scope not in previous_scopes:
                raise OAuthInvalidError(
                    error="invalid_scope",
                    explanation="Invalid scope parameter in request")

        return True