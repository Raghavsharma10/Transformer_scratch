def commit(self, sha):
        """Get a single (repo) commit. See :func:`git_commit` for the Git Data
        Commit.

        :param str sha: (required), sha of the commit
        :returns: :class:`RepoCommit <github3.repos.commit.RepoCommit>` if
            successful, otherwise None
        """
        url = self._build_url('commits', sha, base_url=self._api)
        json = self._json(self._get(url), 200)
        return RepoCommit(json, self) if json else None