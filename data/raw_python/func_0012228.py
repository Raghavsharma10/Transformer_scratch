def set_credentials(self, client_id=None, client_secret=None):
        """
        set given credentials and reset the session
        """
        self._client_id = client_id
        self._client_secret = client_secret

        # make sure to reset session due to credential change
        self._session = None