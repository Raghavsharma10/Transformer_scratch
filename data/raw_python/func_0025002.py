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

        # Add environment variable to manifest
        varname = predix.config.set_env_value(self.use_class, 'uri',
                self._get_uri())
        manifest.add_env_var(varname, self._get_uri())

        manifest.write_manifest()