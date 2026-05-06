def repository(self, owner, repository):
        """Returns a Repository object for the specified combination of
        owner and repository

        :param str owner: (required)
        :param str repository: (required)
        :returns: :class:`Repository <github3.repos.Repository>`
        """
        json = None
        if owner and repository:
            url = self._build_url('repos', owner, repository)
            json = self._json(self._get(url), 200)
        return Repository(json, self) if json else None