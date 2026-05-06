def login(self):
        """
        Login to the RETS board and return an instance of Bulletin
        :return: Bulletin instance
        """
        response = self._request('Login')
        parser = OneXLogin()
        parser.parse(response)

        self.session_id = response.cookies.get(self.session_id_cookie_name, '')

        if parser.headers.get('RETS-Version') is not None:
            self.version = str(parser.headers.get('RETS-Version'))
            self.client.headers['RETS-Version'] = self.version

        for k, v in parser.capabilities.items():
            self.add_capability(k, v)

        if self.capabilities.get('Action'):
            self._request('Action')
        return True