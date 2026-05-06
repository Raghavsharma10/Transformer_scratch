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
        manifest.add_env_var(self.__module__ + '.uri',
                self.service.settings.data['url'])
        manifest.add_env_var(self.__module__ + '.zone_id',
                self.get_predix_zone_id())

        manifest.write_manifest()