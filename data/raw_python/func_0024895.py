def _get_access_key_id(self):
        """
        Returns the access key for an instance of Blob Store service from
        environment inspection.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            return services['predix-blobstore'][0]['credentials']['access_key_id']
        else:
            return predix.config.get_env_value(self, 'access_key_id')