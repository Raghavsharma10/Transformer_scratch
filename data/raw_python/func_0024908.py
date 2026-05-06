def _get_service_keys(self, service_name):
        """
        Return the service keys for the given service.
        """
        guid = self.get_instance_guid(service_name)
        uri = "/v2/service_instances/%s/service_keys" % (guid)
        return self.api.get(uri)