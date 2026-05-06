def asset(self, id):
        """Returns a single Asset.

        :param int id: (required), id of the asset
        :returns: :class:`Asset <github3.repos.release.Asset>`
        """
        data = None
        if int(id) > 0:
            url = self._build_url('releases', 'assets', str(id),
                                  base_url=self._api)
            data = self._json(self._get(url, headers=Release.CUSTOM_HEADERS),
                              200)
        return Asset(data, self) if data else None