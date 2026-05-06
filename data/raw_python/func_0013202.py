def get_log_events(self, login=None, **kwargs):
        """Get a user's log events.

        :param str login: User's login (Default: self._login)
        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login
        )
        log_events_url = GSA_EVENTS_URL.format(login=_login)
        return self._request_api(url=log_events_url).json()