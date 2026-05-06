def set_token(self, redirecturl):
        """Set the token on the session

        :param redirecturl: the original full redirect url
        :type redirecturl: :class:`str`
        :returns: None
        :rtype: None
        :raises: None
        """
        log.debug('Setting the token on %s.' % self.session)
        self.session.token_from_fragment(redirecturl)