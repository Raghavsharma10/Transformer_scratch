def _client_builder(self):
        """Build Elasticsearch client."""
        client_config = self.app.config.get('SEARCH_CLIENT_CONFIG') or {}
        client_config.setdefault(
            'hosts', self.app.config.get('SEARCH_ELASTIC_HOSTS'))
        client_config.setdefault('connection_class', RequestsHttpConnection)
        return Elasticsearch(**client_config)