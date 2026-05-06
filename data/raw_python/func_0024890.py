def create(self):
        """
        Create an instance of the Time Series Service with the typical
        starting settings.
        """
        self.service.create()

        os.environ[predix.config.get_env_key(self.use_class, 'host')] = self.get_eventhub_host()
        os.environ[predix.config.get_env_key(self.use_class, 'port')] = self.get_eventhub_grpc_port()
        os.environ[predix.config.get_env_key(self.use_class, 'wss_publish_uri')] = self.get_publish_wss_uri()
        os.environ[predix.config.get_env_key(self.use_class, 'zone_id')] = self.get_zone_id()