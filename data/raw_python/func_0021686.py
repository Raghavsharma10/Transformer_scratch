def cr_login(self, response):
        """
        Login using email/username and password, used to get the auth token

        @param str account
        @param str password
        @param str hash_id (optional)
        """
        self._state_params['auth'] = response['auth']
        self._user_data = response['user']
        if not self.logged_in:
            raise ApiLoginFailure(response)