def _set_token_expiration_time(self, expires_in):
        """
        Saves the token expiration time by adding the 'expires in' parameter
        to the current datetime (in utc).

        Parameters
        ----------
        expires_in : int
            number of seconds from the time of the request until expiration

        Returns
        -------
        nothing
            saves expiration time in self.token_expiration_time as
            datetime.datetime
        """
        self.token_expiration_time = dt.datetime.utcnow() + \
            dt.timedelta(0, expires_in)