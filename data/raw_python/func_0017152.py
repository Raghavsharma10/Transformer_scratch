def add_labels(self, *args):
        """Add labels to this issue.

        :param str args: (required), names of the labels you wish to add
        :returns: list of :class:`Label`\ s
        """
        url = self._build_url('labels', base_url=self._api)
        json = self._json(self._post(url, data=args), 200)
        return [Label(l, self) for l in json] if json else []