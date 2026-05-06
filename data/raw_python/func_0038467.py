def renew_session(self):
        """
        Clears all session data and starts a new session using the same settings as before.

        This method can be used to clear session data, e.g., cookies. Future requests will use a new session initiated
        with the same settings and authentication method.
        """
        logger.debug("API session renewed")
        self.session = self.authentication.get_session()
        self.session.headers.update({
            'User-Agent': 'MoneyBird for Python %s' % VERSION,
            'Accept': 'application/json',
        })