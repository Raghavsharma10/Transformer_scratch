def tree(self, sha):
        """Get a tree.

        :param str sha: (required), sha of the object for this tree
        :returns: :class:`Tree <github3.git.Tree>`
        """
        json = None
        if sha:
            url = self._build_url('git', 'trees', sha, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Tree(json, self) if json else None