def is_assignee(self, login):
        """Check if the user is a possible assignee for an issue on this
        repository.

        :returns: :class:`bool`
        """
        if not login:
            return False
        url = self._build_url('assignees', login, base_url=self._api)
        return self._boolean(self._get(url), 204, 404)