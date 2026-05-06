def __get_auth(self):
        '''Return auth headers. Will use API Keys if present in settings.'''
        if self.api_key == None and self.login == None:
            self.logger.error("No authentication provided! Unable to connect.")
            sys.exit(1)

        if self.api_key == None:
            self.logger.info("Authenticating with email/password.")
            return [
                "email: " + self.login,
                "password: " + self.password
            ]
        else:
            self.logger.info("Authenticating with API Key.")
            # To auth to the WS using an API key, we generate a signature of a nonce and
            # the WS API endpoint.
            nonce = generate_nonce()
            return [
                "api-nonce: " + str(nonce),
                "api-signature: " + generate_signature(self.api_secret, 'GET', '/realtime', nonce, ''),
                "api-key:" + self.api_key
            ]