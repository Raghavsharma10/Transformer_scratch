def _get_service_config(self):
        """
        Will get configuration for the service from a service key.
        """
        key = self._get_or_create_service_key()

        config = {}
        config['service_key'] = [{'name': self.name}]
        config.update(key['entity']['credentials'])

        return config