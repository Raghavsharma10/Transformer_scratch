def get_url(self, **kwargs):
        """
        Return an url, relative to the request associated with this
        table. Any keywords arguments provided added to the query
        string, replacing existing values.
        """

        return build(
            self._request.path,
            self._request.GET,
            self._meta.prefix,
            **kwargs )