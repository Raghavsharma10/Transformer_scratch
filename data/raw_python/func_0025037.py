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
        uri = predix.config.get_env_key(self.use_class, 'ingest_uri')
        manifest.add_env_var(uri, self.get_ingest_uri())

        zone_id = predix.config.get_env_key(self.use_class, 'ingest_zone_id')
        manifest.add_env_var(zone_id, self.get_ingest_zone_id())

        uri = predix.config.get_env_key(self.use_class, 'query_uri')
        manifest.add_env_var(uri, self.get_query_uri())

        zone_id = predix.config.get_env_key(self.use_class, 'query_zone_id')
        manifest.add_env_var(zone_id, self.get_query_zone_id())

        manifest.write_manifest()