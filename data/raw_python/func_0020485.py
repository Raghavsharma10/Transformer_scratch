def get_user(self, username="~"):
        """
        get info about user (if no user specified, use the one initiating request)

        :param username: str, name of user to get info about, default="~"
        :return: dict
        """
        url = self._build_url("users/%s/" % username, _prepend_namespace=False)
        response = self._get(url)
        check_response(response)
        return response