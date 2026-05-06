def create(self, publish):
        """
        Creates a new publish group.
        """
        target_url = self.client.get_url('PUBLISH', 'POST', 'create')
        r = self.client.request('POST', target_url, json=publish._serialize())
        return self.create_from_result(r.json())