def iter_pages_builds(self, number=-1, etag=None):
        """Iterate over pages builds of this repository.

        :returns: generator of :class:`PagesBuild
            <github3.repos.pages.PagesBuild>`
        """
        url = self._build_url('pages', 'builds', base_url=self._api)
        return self._iter(int(number), url, PagesBuild, etag=etag)