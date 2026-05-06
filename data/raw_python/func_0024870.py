def add_service(self, service_name):
        """
        Add the given service to the manifest.
        """
        if service_name not in self.manifest['services']:
            self.manifest['services'].append(service_name)