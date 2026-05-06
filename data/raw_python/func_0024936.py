def get_spaces(self):
        """
        Return a flat list of the names for spaces in the organization.
        """
        self.spaces = []
        for resource in self._get_spaces()['resources']:
            self.spaces.append(resource['entity']['name'])

        return self.spaces