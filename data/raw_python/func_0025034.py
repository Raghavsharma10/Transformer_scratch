def create(self):
        """
        Create an instance of the Time Series Service with the typical
        starting settings.
        """
        self.service.create()

        predix.config.set_env_value(self.use_class, 'ingest_uri',
                self.get_ingest_uri())
        predix.config.set_env_value(self.use_class, 'ingest_zone_id',
                self.get_ingest_zone_id())

        predix.config.set_env_value(self.use_class, 'query_uri',
                self.get_query_uri())
        predix.config.set_env_value(self.use_class, 'query_zone_id',
                self.get_query_zone_id())