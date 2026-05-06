def list_units(self):
        """Return the current list of the Units in the fleet cluster

        Yields:
            Unit: The next Unit in the cluster

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """
        for page in self._request('Units.List'):
            for unit in page.get('units', []):
                yield Unit(client=self, data=unit)