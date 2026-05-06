def get_user_orcid(self, user_id, password, redirect_uri):
        """Get the user orcid from authentication process.

        Parameters
        ----------
        :param user_id: string
            The id of the user used for authentication.
        :param password: string
            The user password.
        :param redirect_uri: string
            The redirect uri of the institution.

        Returns
        -------
        :returns: string
            The orcid.
        """
        response = self._authenticate(user_id, password, redirect_uri,
                                      '/authenticate')

        return response['orcid']