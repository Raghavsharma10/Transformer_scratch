def create(self, export):
        """
        Create and start processing a new Export.

        :param Export export: The Export to create.
        :rtype: Export
        """
        target_url = self.client.get_url(self._URL_KEY, 'POST', 'create')
        r = self.client.request('POST', target_url, json=export._serialize())
        return export._deserialize(r.json(), self)