def get_notifications(self, login=None, **kwargs):
        """Get the current notifications of a user.

        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login or self._login
        )
        _notif_url = NOTIF_URL.format(login=_login)
        return self._request_api(url=_notif_url).json()