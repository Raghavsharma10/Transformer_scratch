def pull_request(self, number):
        """Get the pull request indicated by ``number``.

        :param int number: (required), number of the pull request.
        :returns: :class:`PullRequest <github3.pulls.PullRequest>`
        """
        json = None
        if int(number) > 0:
            url = self._build_url('pulls', str(number), base_url=self._api)
            json = self._json(self._get(url), 200)
        return PullRequest(json, self) if json else None