def is_assignee_on(self, login, repository):
        """Checks if this user can be assigned to issues on login/repository.

        :returns: :class:`bool`
        """
        url = self._build_url('repos', login, repository, 'assignees',
                              self.login)
        return self._boolean(self._get(url), 204, 404)