def git_commit(self, sha):
        """Get a single (git) commit.

        :param str sha: (required), sha of the commit
        :returns: :class:`Commit <github3.git.Commit>` if successful,
            otherwise None
        """
        json = {}
        if sha:
            url = self._build_url('git', 'commits', sha, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Commit(json, self) if json else None