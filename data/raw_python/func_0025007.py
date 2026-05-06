def _get_uaa_uri(self):
        """
        Returns the URI endpoint for an instance of a UAA
        service instance from environment inspection.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            predix_uaa = services['predix-uaa'][0]['credentials']
            return predix_uaa['uri']
        else:
            return predix.config.get_env_value(self, 'uri')