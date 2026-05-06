def list_manga_series(self, filter=None, content_type='jp_manga'):
        """Get a list of manga series
        """

        result = self._manga_api.list_series(filter, content_type)
        return result