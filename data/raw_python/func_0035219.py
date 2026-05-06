def get_session(self, redirect_url):
        """Complete the Authorization Code Grant process.
        The redirect URL received after the user has authorized
        your application contains an authorization code. Use this
        authorization code to request an access token.
        Parameters
            redirect_url (str)
                The full URL that the Lyft server redirected to after
                the user authorized your app.
        Returns
            (Session)
                A Session object with OAuth 2.0 credentials.
        """
        query_params = self._extract_query(redirect_url)
        authorization_code = self._verify_query(query_params)

        response = request_access_token(
            grant_type=auth.AUTHORIZATION_CODE_GRANT,
            client_id=self.client_id,
            client_secret=self.client_secret,
            code=authorization_code,
        )

        oauth2credential = OAuth2Credential.make_from_response(
            response=response,
            grant_type=auth.AUTHORIZATION_CODE_GRANT,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

        return Session(oauth2credential=oauth2credential)