def add_to_manifest(self, manifest):
        """
        Add useful details to the manifest about this service
        so that it can be used in an application.

        :param manifest: An predix.admin.app.Manifest object
            instance that manages reading/writing manifest config
            for a cloud foundry app.
        """
        # Add this service to the list of services
        manifest.add_service(self.service.name)

        # Add environment variables

        url = predix.config.get_env_key(self.use_class, 'url')
        manifest.add_env_var(url, self.service.settings.data['url'])

        akid = predix.config.get_env_key(self.use_class, 'access_key_id')
        manifest.add_env_var(akid, self.service.settings.data['access_key_id'])

        bucket = predix.config.get_env_key(self.use_class, 'bucket_name')
        manifest.add_env_var(bucket, self.service.settings.data['bucket_name'])

        host = predix.config.get_env_key(self.use_class, 'host')
        manifest.add_env_var(host, self.service.settings.data['host'])

        secret_access_key = predix.config.get_env_key(self.use_class, 'secret_access_key')
        manifest.add_env_var(secret_access_key, self.service.settings.data['secret_access_key'])

        manifest.write_manifest()