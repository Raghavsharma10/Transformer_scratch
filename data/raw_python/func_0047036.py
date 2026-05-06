def authenticate(self):
        """Set BitMEX authentication information."""
        if self.apiKey:
            return
        loginResponse = self._curl_bitmex(
            api="user/login",
            postdict={'email': self.login, 'password': self.password, 'token': self.otpToken})
        self.token = loginResponse['id']
        self.session.headers.update({'access-token': self.token})