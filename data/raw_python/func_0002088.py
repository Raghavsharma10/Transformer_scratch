def create(self, set):
        """
        Creates a new Set.
        """
        target_url = self.client.get_url('SET', 'POST', 'create')
        r = self.client.request('POST', target_url, json=set._serialize())
        return set._deserialize(r.json(), self)