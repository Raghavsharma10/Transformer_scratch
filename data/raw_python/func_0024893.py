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
        manifest.add_env_var(predix.config.get_env_key(self.use_class, 'host'), self.get_eventhub_host())
        manifest.add_env_var(predix.config.get_env_key(self.use_class, 'port'), self.get_eventhub_grpc_port())
        manifest.add_env_var(predix.config.get_env_key(self.use_class, 'wss_publish_uri'), self.get_publish_wss_uri())
        manifest.add_env_var(predix.config.get_env_key(self.use_class, 'zone_id'), self.get_zone_id())

        manifest.write_manifest()