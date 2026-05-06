def update(self, sha, force=False):
        """Update this reference.

        :param str sha: (required), sha of the reference
        :param bool force: (optional), force the update or not
        :returns: bool

        """
        data = {'sha': sha, 'force': force}
        json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_(json)
            return True
        return False