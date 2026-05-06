def get_apps(self):
        """
        Returns a flat list of the names for the apps in
        the organization.
        """
        apps = []
        for resource in self._get_apps()['resources']:
            apps.append(resource['entity']['name'])

        return apps