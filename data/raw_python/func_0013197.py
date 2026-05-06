def get_picture(self, login=None, **kwargs):
        """Get a user's picture.

        :param str login: Login of the user to check
        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login or self._login
        )
        _activities_url = PICTURE_URL.format(login=_login)
        return self._request_api(url=_activities_url).content