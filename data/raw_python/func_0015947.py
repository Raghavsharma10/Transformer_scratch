def authenticate(self, username, password):
        """
        Uses a Smappee username and password to request an access token,
        refresh token and expiry date.

        Parameters
        ----------
        username : str
        password : str

        Returns
        -------
        requests.Response
            access token is saved in self.access_token
            refresh token is saved in self.refresh_token
            expiration time is set in self.token_expiration_time as
            datetime.datetime
        """
        url = URLS['token']
        data = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": username,
            "password": password
        }
        r = requests.post(url, data=data)
        r.raise_for_status()
        j = r.json()
        self.access_token = j['access_token']
        self.refresh_token = j['refresh_token']
        self._set_token_expiration_time(expires_in=j['expires_in'])
        return r