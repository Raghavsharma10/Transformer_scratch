def execute_query(self, payload):
        '''Execute the query and returns and response'''
        if vars(self).get('oauth'):
            if not self.oauth.token_is_valid(): # Refresh token if token has expired
                self.oauth.refresh_token()
            response = self.oauth.session.get(self.PRIVATE_URL, params= payload, header_auth=True)
        else:
            response = requests.get(self.PUBLIC_URL, params= payload)

        self._response = response # Saving last response object.
        return response