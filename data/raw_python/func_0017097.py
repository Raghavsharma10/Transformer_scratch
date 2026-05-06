def label(self, name):
        """Get the label specified by ``name``

        :param str name: (required), name of the label
        :returns: :class:`Label <github3.issues.label.Label>` if successful,
            else None
        """
        json = None
        if name:
            url = self._build_url('labels', name, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Label(json, self) if json else None