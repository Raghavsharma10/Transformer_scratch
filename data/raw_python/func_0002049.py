def start_scan(self, source_id):
        """
        Start a new scan of a Source.

        :rtype: Scan
        """
        target_url = self.client.get_url('SCAN', 'POST', 'create', {'source_id': source_id})
        r = self.client.request('POST', target_url, json={})
        return self.client.get_manager(Scan).create_from_result(r.json())