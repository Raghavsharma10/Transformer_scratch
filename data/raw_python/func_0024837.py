def add_to_manifest(self, manifest):
        """
        Add useful details to the manifest about this service
        so that it can be used in an application.

        :param manifest: An predix.admin.app.Manifest object
            instance that manages reading/writing manifest config
            for a cloud foundry app.
        """
        # Add this service to list of services
        manifest.add_service(self.service.name)

        # Add environment variables
        zone_id = predix.config.get_env_key(self.use_class, 'zone_id')
        manifest.add_env_var(zone_id,
                self.service.settings.data['zone']['http-header-value'])

        uri = predix.config.get_env_key(self.use_class, 'uri')
        manifest.add_env_var(uri, self.service.settings.data['uri'])

        manifest.write_manifest()