def start_import(self, layer_id, version_id):
        """
        Starts importing the specified draft version (cancelling any running import),
        even if the data object hasn’t changed from the previous version.
        """
        target_url = self.client.get_url('VERSION', 'POST', 'import', {'layer_id': layer_id, 'version_id': version_id})
        r = self.client.request('POST', target_url, json={})
        return self.create_from_result(r.json())