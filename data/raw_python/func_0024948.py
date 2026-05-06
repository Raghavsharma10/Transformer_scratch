def _get_zone_id(self):
        """
        Returns the Predix Zone Id for the service that is a required
        header in service calls.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            predix_asset = services['predix-asset'][0]['credentials']
            return predix_asset['zone']['http-header-value']
        else:
            return predix.config.get_env_value(self, 'zone_id')