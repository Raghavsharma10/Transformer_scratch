def _get_host(self):
        """
        Returns the host address for an instance of Blob Store service from
        environment inspection.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            host = services['predix-blobstore'][0]['credentials']['host']
        else:
            host = predix.config.get_env_value(self, 'host')

        # Protocol may not always be included in host setting
        if 'https://' not in host:
            host = 'https://' + host

        return host