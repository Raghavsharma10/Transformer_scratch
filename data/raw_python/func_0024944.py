def add_to_manifest(self, manifest):
        """
        Add useful details to the manifest about this service so
        that it can be used in an application.

        :param manifest: A predix.admin.app.Manifest object instance
            that manages reading/writing manifest config for a
            cloud foundry app.
        """
        manifest.add_service(self.service.name)

        hostname = predix.config.get_env_key(self.use_class, 'hostname')
        manifest.add_env_var(hostname, self.service.settings.data['hostname'])

        password = predix.config.get_env_key(self.use_class, 'password')
        manifest.add_env_var(password, self.service.settings.data['password'])

        port = predix.config.get_env_key(self.use_class, 'port')
        manifest.add_env_var(port, self.service.settings.data['port'])

        username = predix.config.get_env_key(self.use_class, 'username')
        manifest.add_env_var(username, self.service.settings.data['username'])

        uri = predix.config.get_env_key(self.use_class, 'uri')
        manifest.add_env_var(uri, self.service.settings.data['uri'])

        manifest.write_manifest()