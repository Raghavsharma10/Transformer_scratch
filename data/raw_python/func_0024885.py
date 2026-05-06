def _get_or_create_service_key(self):
        """
        Get a service key or create one if needed.
        """
        keys = self.service._get_service_keys(self.name)
        for key in keys['resources']:
            if key['entity']['name'] == self.service_name:
                return self.service.get_service_key(self.name,
                        self.service_name)

        self.service.create_service_key(self.name, self.service_name)
        return self.service.get_service_key(self.name, self.service_name)