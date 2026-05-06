def login(self, email=None, password=None, timeout=5):
        """Log in to CANDY HOUSE account. Return True on success."""
        if email is not None:
            self.email = email
        if password is not None:
            self.password = password

        url = self.api_url + API_LOGIN_ENDPOINT
        data = json.dumps({'email': self.email, 'password': self.password})
        headers = {'Content-Type': 'application/json'}
        response = None

        try:
            response = self.session.post(url, data=data, headers=headers,
                                         timeout=timeout)
        except requests.exceptions.ConnectionError:
            _LOGGER.warning("Unable to connect to %s", url)
        except requests.exceptions.Timeout:
            _LOGGER.warning("No response from %s", url)

        if response is not None:
            if response.status_code == 200:
                self.auth_token = json.loads(response.text)['authorization']
                return True
            else:
                _LOGGER.warning("Login failed for %s: %s", self.email,
                                response.text)
        else:
            _LOGGER.warning("Login failed for %s", self.email)

        return False