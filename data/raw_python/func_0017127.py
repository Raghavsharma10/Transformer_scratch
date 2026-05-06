def release(self, id):
        """Get a single release.

        :param int id: (required), id of release
        :returns: :class:`Release <github3.repos.release.Release>`
        """
        json = None
        if int(id) > 0:
            url = self._build_url('releases', str(id), base_url=self._api)
            json = self._json(self._get(url), 200)
        return Release(json, self) if json else None