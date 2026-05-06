def is_merged(self):
        """Checks to see if the pull request was merged.

        :returns: bool
        """
        url = self._build_url('merge', base_url=self._api)
        return self._boolean(self._get(url), 204, 404)