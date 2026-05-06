def login(self, password):
        """Login to filemail as the current user.

        :param password:
        :type password: ``str``
        """

        method, url = get_URL('login')
        payload = {
            'apikey': self.config.get('apikey'),
            'username': self.username,
            'password': password,
            'source': 'Desktop'
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return True

        hellraiser(res)