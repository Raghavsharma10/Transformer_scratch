def get_instance(self, service_name):
        """
        Retrieves a service instance with the given name.
        """
        for resource in self.space._get_instances():
            if resource['entity']['name'] == service_name:
                return resource['entity']