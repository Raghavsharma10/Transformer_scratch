def create_commit(self, message, tree, parents, author={}, committer={}):
        """Create a commit on this repository.

        :param str message: (required), commit message
        :param str tree: (required), SHA of the tree object this
            commit points to
        :param list parents: (required), SHAs of the commits that were parents
            of this commit. If empty, the commit will be written as the root
            commit.  Even if there is only one parent, this should be an
            array.
        :param dict author: (optional), if omitted, GitHub will
            use the authenticated user's credentials and the current
            time. Format: {'name': 'Committer Name', 'email':
            'name@example.com', 'date': 'YYYY-MM-DDTHH:MM:SS+HH:00'}
        :param dict committer: (optional), if ommitted, GitHub will use the
            author parameters. Should be the same format as the author
            parameter.
        :returns: :class:`Commit <github3.git.Commit>` if successful, else
            None
        """
        json = None
        if message and tree and isinstance(parents, list):
            url = self._build_url('git', 'commits', base_url=self._api)
            data = {'message': message, 'tree': tree, 'parents': parents,
                    'author': author, 'committer': committer}
            json = self._json(self._post(url, data=data), 201)
        return Commit(json, self) if json else None