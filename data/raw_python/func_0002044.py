def list_datasources(self, source_id):
        """
        Filterable list of Datasources for a Source.
        """
        target_url = self.client.get_url('DATASOURCE', 'GET', 'multi', {'source_id': source_id})
        return base.Query(self.client.get_manager(Datasource), target_url)