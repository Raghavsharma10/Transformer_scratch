def init_session(self, get_token=True):
        """
        init a new oauth2 session that is required to access the cloud

        :param bool get_token: if True, a token will be obtained, after
                               the session has been created
        """
        if (self._client_id is None) or (self._client_secret is None):
            sys.exit(
                "Please make sure to set the client id and client secret "
                "via the constructor, the environment variables or the config "
                "file; otherwise, the LaMetric cloud cannot be accessed. "
                "Abort!"
            )

        self._session = OAuth2Session(
            client=BackendApplicationClient(client_id=self._client_id)
        )

        if get_token is True:
            # get oauth token
            self.get_token()