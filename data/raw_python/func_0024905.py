def get_instance_guid(self, service_name):
        """
        Returns the GUID for the service instance with
        the given name.
        """
        summary = self.space.get_space_summary()
        for service in summary['services']:
            if service['name'] == service_name:
                return service['guid']

        raise ValueError("No service with name '%s' found." % (service_name))