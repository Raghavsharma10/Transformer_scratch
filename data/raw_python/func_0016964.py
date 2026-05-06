def update(self, body):
        """Update this comment.

        :param str body: (required)
        :returns: bool
        """
        json = None
        if body:
            json = self._json(self._post(self._api, data={'body': body}), 200)

        if json:
            self._update_(json)
            return True
        return False