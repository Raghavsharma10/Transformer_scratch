def update(self, scopes=[], add_scopes=[], rm_scopes=[], note='',
               note_url=''):
        """Update this authorization.

        :param list scopes: (optional), replaces the authorization scopes with
            these
        :param list add_scopes: (optional), scopes to be added
        :param list rm_scopes: (optional), scopes to be removed
        :param str note: (optional), new note about authorization
        :param str note_url: (optional), new note URL about this authorization
        :returns: bool

        """
        success = False
        json = None
        if scopes:
            d = {'scopes': scopes}
            json = self._json(self._post(self._api, data=d), 200)
        if add_scopes:
            d = {'add_scopes': add_scopes}
            json = self._json(self._post(self._api, data=d), 200)
        if rm_scopes:
            d = {'remove_scopes': rm_scopes}
            json = self._json(self._post(self._api, data=d), 200)
        if note or note_url:
            d = {'note': note, 'note_url': note_url}
            json = self._json(self._post(self._api, data=d), 200)

        if json:
            self._update_(json)
            success = True

        return success