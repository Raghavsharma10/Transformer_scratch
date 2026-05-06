def pages(self):
        """Get information about this repository's pages site.

        :returns: :class:`PagesInfo <github3.repos.pages.PagesInfo>`
        """
        url = self._build_url('pages', base_url=self._api)
        json = self._json(self._get(url), 200)
        return PagesInfo(json) if json else None