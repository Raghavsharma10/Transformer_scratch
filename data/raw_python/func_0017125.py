def readme(self):
        """Get the README for this repository.

        :returns: :class:`Contents <github3.repos.contents.Contents>`
        """
        url = self._build_url('readme', base_url=self._api)
        json = self._json(self._get(url), 200)
        return Contents(json, self) if json else None