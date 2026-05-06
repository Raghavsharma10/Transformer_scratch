def get_token(self, user_id, password, redirect_uri,
                  scope='/activities/update'):
        """Get the token.

        Parameters
        ----------
        :param user_id: string

            The id of the user used for authentication.
        :param password: string
            The user password.
        :param redirect_uri: string
            The redirect uri of the institution.
        :param scope: string
            The desired scope. For example '/activities/update',
            '/read-limited', etc.

        Returns
        -------
        :returns: string
            The token.
        """
        return super(MemberAPI, self).get_token(user_id, password,
                                                redirect_uri, scope)