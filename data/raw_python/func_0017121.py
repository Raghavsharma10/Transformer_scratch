def merge(self, base, head, message=''):
        """Perform a merge from ``head`` into ``base``.

        :param str base: (required), where you're merging into
        :param str head: (required), where you're merging from
        :param str message: (optional), message to be used for the commit
        :returns: :class:`RepoCommit <github3.repos.commit.RepoCommit>`
        """
        url = self._build_url('merges', base_url=self._api)
        data = {'base': base, 'head': head}
        if message:
            data['commit_message'] = message
        json = self._json(self._post(url, data=data), 201)
        return RepoCommit(json, self) if json else None