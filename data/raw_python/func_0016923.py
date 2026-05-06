def update(self, title=None, body=None, state=None):
        """Update this pull request.

        :param str title: (optional), title of the pull
        :param str body: (optional), body of the pull request
        :param str state: (optional), ('open', 'closed')
        :returns: bool
        """
        data = {'title': title, 'body': body, 'state': state}
        json = None
        self._remove_none(data)

        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)

        if json:
            self._update_(json)
            return True
        return False