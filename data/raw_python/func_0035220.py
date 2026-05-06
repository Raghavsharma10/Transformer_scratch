def get_session(self):
        """Create Session to store credentials.
        Returns
            (Session)
                A Session object with OAuth 2.0 credentials.
        """
        response = request_access_token(
            grant_type=auth.CLIENT_CREDENTIAL_GRANT,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=self.scopes,
        )

        oauth2credential = OAuth2Credential.make_from_response(
            response=response,
            grant_type=auth.CLIENT_CREDENTIAL_GRANT,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

        return Session(oauth2credential=oauth2credential)