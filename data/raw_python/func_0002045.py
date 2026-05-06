def get_datasource(self, source_id, datasource_id):
        """
        Get a Datasource object

        :rtype: Datasource
        """
        target_url = self.client.get_url('DATASOURCE', 'GET', 'single', {'source_id': source_id, 'datasource_id': datasource_id})
        return self.client.get_manager(Datasource)._get(target_url)