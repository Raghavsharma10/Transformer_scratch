def create(self, **kwargs):
        """
        Create an instance of the Blob Store Service with the typical
        starting settings.
        """
        self.service.create(**kwargs)

        predix.config.set_env_value(self.use_class, 'url',
                self.service.settings.data['url'])
        predix.config.set_env_value(self.use_class, 'access_key_id',
                self.service.settings.data['access_key_id'])
        predix.config.set_env_value(self.use_class, 'bucket_name',
                self.service.settings.data['bucket_name'])
        predix.config.set_env_value(self.use_class, 'host',
                self.service.settings.data['host'])
        predix.config.set_env_value(self.use_class, 'secret_access_key',
                self.service.settings.data['secret_access_key'])