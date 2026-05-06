def is_starred(self):
        """Check to see if this gist is starred by the authenticated user.

        :returns: bool -- True if it is starred, False otherwise

        """
        url = self._build_url('star', base_url=self._api)
        return self._boolean(self._get(url), 204, 404)