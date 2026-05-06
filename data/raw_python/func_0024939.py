def get_services(self):
        """
        Returns a flat list of the service names available
        from the marketplace for this space.
        """
        services = []
        for resource in self._get_services()['resources']:
            services.append(resource['entity']['label'])

        return services