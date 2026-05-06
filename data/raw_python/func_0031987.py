def wrap_get_user(cls, response):
        """Wrap the response from getting a user into an instance
        and return it

        :param response: The response from getting a user
        :type response: :class:`requests.Response`
        :returns: the new user instance
        :rtype: :class:`list` of :class:`User`
        :raises: None
        """
        json = response.json()
        u = cls.wrap_json(json)
        return u