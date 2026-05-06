def login(self, email=None, password=None, app_id=None, api_key=None):
        """Login to MediaFire account.

        Keyword arguments:
        email -- account email
        password -- account password
        app_id -- application ID
        api_key -- API Key (optional)
        """
        session_token = self.api.user_get_session_token(
            app_id=app_id, email=email, password=password, api_key=api_key)

        # install session token back into api client
        self.api.session = session_token