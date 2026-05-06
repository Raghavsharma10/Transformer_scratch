def get_service_key(self, service_name, key_name):
        """
        Returns the service key details.

        Similar to `cf service-key`.
        """
        for key in self._get_service_keys(service_name)['resources']:
            if key_name == key['entity']['name']:
                guid = key['metadata']['guid']

                uri = "/v2/service_keys/%s" % (guid)
                return self.api.get(uri)

        return None