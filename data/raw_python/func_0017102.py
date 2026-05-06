def iter_commits(self, sha=None, path=None, author=None, number=-1,
                     etag=None, since=None, until=None):
        """Iterate over commits in this repository.

        :param str sha: (optional), sha or branch to start listing commits
            from
        :param str path: (optional), commits containing this path will be
            listed
        :param str author: (optional), GitHub login, real name, or email to
            filter commits by (using commit author)
        :param int number: (optional), number of commits to return. Default:
            -1 returns all commits
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :param since: (optional), Only commits after this date will
            be returned. This can be a `datetime` or an `ISO8601` formatted
            date string.
        :type since: datetime or string
        :param until: (optional), Only commits before this date will
            be returned. This can be a `datetime` or an `ISO8601` formatted
            date string.
        :type until: datetime or string

        :returns: generator of
            :class:`RepoCommit <github3.repos.commit.RepoCommit>`\ s
        """
        params = {'sha': sha, 'path': path, 'author': author,
                  'since': timestamp_parameter(since),
                  'until': timestamp_parameter(until)}

        self._remove_none(params)
        url = self._build_url('commits', base_url=self._api)
        return self._iter(int(number), url, RepoCommit, params, etag)