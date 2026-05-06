def delete(self):
        """Users with push access to the repository can delete a release.

        :returns: True if successful; False if not successful
        """
        url = self._api
        return self._boolean(
            self._delete(url, headers=Release.CUSTOM_HEADERS),
            204,
            404
        )