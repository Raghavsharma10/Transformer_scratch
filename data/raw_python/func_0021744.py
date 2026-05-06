def login(self, username, password):
        """Login with the given username/email and password

        Calling this method is not required if credentials were provided in
        the constructor, but it could be used to switch users or something maybe

        @return bool
        """
        # we could get stuck in an inconsistent state if got an exception while
        # trying to login with different credentials than what is stored so
        # we rollback the state to prevent that
        state_snapshot = self._state.copy()
        try:
            self._ajax_api.User_Login(name=username, password=password)
            self._android_api.login(account=username, password=password)
            self._manga_api.cr_login(account=username, password=password)
        except Exception as err:
            # something went wrong, rollback
            self._state = state_snapshot
            raise err
        self._state['username'] = username
        self._state['password'] = password
        return self.logged_in