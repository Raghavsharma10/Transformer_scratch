def add_oauth_header(self):
        """
        Validate token and add the proper header for further requests.
        :return: (None)
        """
        # abort if no token
        oauth_token = self._get_token()
        if not oauth_token:
            return

        # add oauth header & reach the api
        self.headers["Authorization"] = "token " + oauth_token
        url = self._api_url("user")
        raw_resp = self.requests.get(url)
        resp = raw_resp.json()

        # abort & remove header if token is invalid
        if resp.get("login", None) != self.user:
            self.oops("Invalid token for user " + self.user)
            self.headers.pop("Authorization")
            return

        self.is_authenticated = True
        self.yeah("User {} authenticated".format(self.user))