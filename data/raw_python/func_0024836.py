def create(self):
        """
        Create an instance of the US Weather Forecast Service with
        typical starting settings.
        """
        self.service.create()

        # Set env vars for immediate use
        zone_id = predix.config.get_env_key(self.use_class, 'zone_id')
        zone = self.service.settings.data['zone']['http-header-value']
        os.environ[zone_id] = zone

        uri = predix.config.get_env_key(self.use_class, 'uri')
        os.environ[uri] = self.service.settings.data['uri']