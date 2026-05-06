def authenticate(self, email=None, password=None):
        """
        Attempt to authenticate the user.

        Parameters
        ----------
        email : string
            The email of a user on Lending Club
        password : string
            The user's password, for authentication.

        Returns
        -------
        boolean
            True if the user authenticated or raises an exception if not

        Raises
        ------
        session.AuthenticationError
            If authentication failed
        session.NetworkError
            If a network error occurred
        """
        if self.session.authenticate(email, password):
            return True