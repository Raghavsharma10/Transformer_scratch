def iter_assets(self, number=-1, etag=None):
        """Iterate over the assets available for this release.

        :param int number: (optional), Number of assets to return
        :param str etag: (optional), last ETag header sent
        :returns: generator of :class:`Asset <Asset>` objects
        """
        url = self._build_url('assets', base_url=self._api)
        return self._iter(number, url, Asset, etag=etag)