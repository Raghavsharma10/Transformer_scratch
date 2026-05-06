def get_events(self, login=None, start_date=None, end_date=None, **kwargs):
        """Get a user's events.

        :param str login: User's login (Default: self._login)
        :param str start_date: Start date
        :param str end_date: To date
        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login
        )
        log_events_url = EVENTS_URL.format(
            login=_login,
            start_date=start_date,
            end_date=end_date,
        )
        return self._request_api(url=log_events_url).json()