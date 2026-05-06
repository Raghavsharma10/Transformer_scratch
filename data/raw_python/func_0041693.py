def login(self) -> bool:
        """
        Authorizes a user and returns a bool value of the result
        """
        response = self.get(self.LOGIN_URL)
        login_url = get_base_url(response.text)
        login_data = {'email': self._login, 'pass': self._password}
        login_response = self.post(login_url, login_data)
        url_params = get_url_params(login_response.url)
        self.check_for_additional_actions(url_params,
                                          login_response.text,
                                          login_data)
        if 'remixsid' in self.cookies or 'remixsid6' in self.cookies:
            return True