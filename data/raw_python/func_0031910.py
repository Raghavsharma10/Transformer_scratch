def do_POST(self, ):
        """Handle POST requests

        When the user is redirected, this handler will respond with a website
        which will send a post request with the url fragment as parameters.
        This will get the parameters and store the original redirection
        url and fragments in :data:`LoginServer.tokenurl`.

        :returns: None
        :rtype: None
        :raises: None
        """
        log.debug('POST')
        self._set_headers()
        # convert the parameters back to the original fragment
        # because we need to send the original uri to set_token
        # url fragments will not show up in self.path though.
        # thats why we make the hassle to send it as a post request.
        # Note: oauth does not allow for http connections
        # but twitch does, so we fake it
        ruri = constants.REDIRECT_URI.replace('http://', 'https://')
        self.server.set_token(ruri + self.path.replace('?', '#'))