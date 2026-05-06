def create_label(self, name, color):
        """Create a label for this repository.

        :param str name: (required), name to give to the label
        :param str color: (required), value of the color to assign to the
            label, e.g., '#fafafa' or 'fafafa' (the latter is what is sent)
        :returns: :class:`Label <github3.issues.label.Label>` if successful,
            else None
        """
        json = None
        if name and color:
            data = {'name': name, 'color': color.strip('#')}
            url = self._build_url('labels', base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
        return Label(json, self) if json else None