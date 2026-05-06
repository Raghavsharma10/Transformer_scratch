def refresh_oauth_credential(self):
        """Refresh session's OAuth 2.0 credentials if they are stale."""
        credential = self.session.oauth2credential

        if credential.is_stale():
            refresh_session = refresh_access_token(credential)
            self.session = refresh_session