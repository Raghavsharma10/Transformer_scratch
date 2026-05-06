def _get_service_bindings(self, service_name):
        """
        Return the service bindings for the service instance.
        """
        instance = self.get_instance(service_name)
        return self.api.get(instance['service_bindings_url'])