def delete_service_bindings(self, service_name):
        """
        Remove service bindings to applications.
        """
        instance = self.get_instance(service_name)
        return self.api.delete(instance['service_bindings_url'])