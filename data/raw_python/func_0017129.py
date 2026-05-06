def tag(self, sha):
        """Get an annotated tag.

        http://learn.github.com/p/tagging.html

        :param str sha: (required), sha of the object for this tag
        :returns: :class:`Tag <github3.git.Tag>`
        """
        json = None
        if sha:
            url = self._build_url('git', 'tags', sha, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Tag(json) if json else None