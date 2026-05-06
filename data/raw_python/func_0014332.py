def url(self, **kwargs):
        """
        Returns a formatted URL for the asset's File
        with serialized parameters.

        Usage:
            >>> my_asset.url()
            "//images.contentful.com/spaces/foobar/..."
            >>> my_asset.url(w=120, h=160)
            "//images.contentful.com/spaces/foobar/...?w=120&h=160"
        """

        url = self.fields(self._locale()).get('file', {}).get('url', '')
        args = ['{0}={1}'.format(k, v) for k, v in kwargs.items()]

        if args:
            url += '?{0}'.format('&'.join(args))

        return url