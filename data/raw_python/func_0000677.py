def authenticate(self):
        """Authenticate the user and setup our state."""
        valid = self._session_check()
        if self._is_authenticated and valid:
            self._log.debug("[!] User has already authenticated")
            return
        init = self._session.get(url=self.LOGIN_URL, headers=self.HEADERS)
        soup = BeautifulSoup(init.content, "html.parser")
        soup_login = soup.find('form').find_all('input')
        post_data = dict()
        for u in soup_login:
            if u.has_attr('name') and u.has_attr('value'):
                post_data[u['name']] = u['value']
        post_data['Email'] = self._email
        post_data['Passwd'] = self._password
        response = self._session.post(url=self.AUTH_URL, data=post_data,
                                      headers=self.HEADERS)
        if self.CAPTCHA_KEY in str(response.content):
            raise AccountCaptcha('Google is forcing a CAPTCHA. To get around this issue, run the google-alerts with the seed option to open an interactive authentication session. Once authenticated, this module will cache your session and load that in the future')
        cookies = [x.name for x in response.cookies]
        if 'SIDCC' not in cookies:
            raise InvalidCredentials("Email or password was incorrect.")
        with open(SESSION_FILE, 'wb') as f:
            cookies = requests.utils.dict_from_cookiejar(self._session.cookies)
            pickle.dump(cookies, f, protocol=2)
            self._log.debug("Saved session to disk for future reference")
        self._log.debug("User successfully authenticated")
        self._is_authenticated = True
        self._process_state()
        return