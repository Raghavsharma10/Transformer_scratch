def list_upcoming(cls):
        """
        Returns a collection of upcoming tasks (tasks that have not yet been completed,
        regardless of whether they’re overdue) for the authenticated user

        :return:
        :rtype: list
        """
        return fields.ListField(name=cls.ENDPOINT, init_class=cls).decode(
            cls.element_from_string(
                cls._get_request(endpoint=cls.ENDPOINT + '/upcoming').text
            )
        )