def create_service_key(self, service_name, key_name):
        """
        Create a service key for the given service.
        """
        if self.has_key(service_name, key_name):
            logging.warning("Reusing existing service key %s" % (key_name))
            return self.get_service_key(service_name, key_name)

        body = {
            'service_instance_guid': self.get_instance_guid(service_name),
            'name': key_name
            }

        return self.api.post('/v2/service_keys', body)