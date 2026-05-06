def replace_labels(self, labels):
        """Replace all labels on this issue with ``labels``.

        :param list labels: label names
        :returns: bool
        """
        url = self._build_url('labels', base_url=self._api)
        json = self._json(self._put(url, data=dumps(labels)), 200)
        return [Label(l, self) for l in json] if json else []