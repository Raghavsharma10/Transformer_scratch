def issue(self, number):
        """Get the issue specified by ``number``.

        :param int number: (required), number of the issue on this repository
        :returns: :class:`Issue <github3.issues.issue.Issue>` if successful,
            otherwise None
        """
        json = None
        if int(number) > 0:
            url = self._build_url('issues', str(number), base_url=self._api)
            json = self._json(self._get(url), 200)
        return Issue(json, self) if json else None