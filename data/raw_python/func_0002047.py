def get_scan(self, source_id, scan_id):
        """
        Get a Scan object

        :rtype: Scan
        """
        target_url = self.client.get_url('SCAN', 'GET', 'single', {'source_id': source_id, 'scan_id': scan_id})
        return self.client.get_manager(Scan)._get(target_url)