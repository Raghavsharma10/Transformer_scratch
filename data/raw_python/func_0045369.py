def me(cls):
        """
        Returns information about the currently authenticated user.

        :return:
        :rtype: User
        """
        return fields.ObjectField(name=cls.ENDPOINT, init_class=cls).decode(
            cls.element_from_string(
                cls._get_request(endpoint=cls.ENDPOINT + '/me').text
            )
        )