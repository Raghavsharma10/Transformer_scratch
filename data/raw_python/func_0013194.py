def get_current_activities(self, login=None, **kwargs):
        """Get the current activities of user.

        Either use the `login` param, or the client's login if unset.
        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login or self._login
        )
        _activity_url = ACTIVITY_URL.format(login=_login)
        return self._request_api(url=_activity_url).json()