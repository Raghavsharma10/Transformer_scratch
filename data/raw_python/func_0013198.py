def get_projects(self, **kwargs):
        """Get a user's project.

        :param str login: User's login (Default: self._login)
        :return: JSON
        """

        _login = kwargs.get('login', self._login)
        search_url = SEARCH_URL.format(login=_login)
        return self._request_api(url=search_url).json()