def compare_commits(self, base, head):
        """Compare two commits.

        :param str base: (required), base for the comparison
        :param str head: (required), compare this against base
        :returns: :class:`Comparison <github3.repos.comparison.Comparison>` if
            successful, else None
        """
        url = self._build_url('compare', base + '...' + head,
                              base_url=self._api)
        json = self._json(self._get(url), 200)
        return Comparison(json) if json else None