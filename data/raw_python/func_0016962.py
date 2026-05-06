def star(self):
        """Star this gist.

        :returns: bool -- True if successful, False otherwise

        """
        url = self._build_url('star', base_url=self._api)
        return self._boolean(self._put(url), 204, 404)