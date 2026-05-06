def get_instances(self):
        """
        Returns a flat list of the names of services created
        in this space.
        """
        services = []
        for resource in self._get_instances():
            services.append(resource['entity']['name'])

        return services