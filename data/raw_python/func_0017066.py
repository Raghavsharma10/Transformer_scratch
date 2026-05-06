def branch(self, name):
        """Get the branch ``name`` of this repository.

        :param str name: (required), branch name
        :type name: str
        :returns: :class:`Branch <github3.repos.branch.Branch>`
        """
        json = None
        if name:
            url = self._build_url('branches', name, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Branch(json, self) if json else None