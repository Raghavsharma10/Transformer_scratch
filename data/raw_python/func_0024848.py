def get_orgs(self):
        """
        Returns a flat list of the names for the organizations
        user belongs.
        """
        orgs = []
        for resource in self._get_orgs()['resources']:
            orgs.append(resource['entity']['name'])

        return orgs