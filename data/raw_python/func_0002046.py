def list_scans(self, source_id=None):
        """
        Filterable list of Scans for a Source.
        Ordered newest to oldest by default
        """
        if source_id:
            target_url = self.client.get_url('SCAN', 'GET', 'multi', {'source_id': source_id})
        else:
            target_url = self.client.get_ulr('SCAN', 'GET', 'all')
        return base.Query(self.client.get_manager(Scan), target_url)