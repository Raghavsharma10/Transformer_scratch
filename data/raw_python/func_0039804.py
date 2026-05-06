def login(self, username, password):
        """Logs into reddit with supplied credentials using SSL.  Returns :class:`requests.Response` object, or raises :class:`exceptions.LoginFail` or :class:`exceptions.BadResponse` if not a 200 response.
        
        URL: ``https://ssl.reddit.com/api/login``
        
        :param username: reddit username
        :param password: corresponding reddit password
        """
        data = dict(user=username, passwd=password, api_type='json')
        r = requests.post(LOGIN_URL, data=data)
        if r.status_code == 200:
            try:
                j = json.loads(r.content)
                self._cookies = r.cookies
                self._modhash = j['json']['data']['modhash']
                self._username = username
                return r
            except Exception:
                raise LoginFail()
        else:
            raise BadResponse(r)