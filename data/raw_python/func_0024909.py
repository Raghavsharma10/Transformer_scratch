def get_service_keys(self, service_name):
        """
        Returns a flat list of the names of the service keys
        for the given service.
        """
        keys = []
        for key in self._get_service_keys(service_name)['resources']:
            keys.append(key['entity']['name'])

        return keys