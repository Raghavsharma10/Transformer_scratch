def get_unit(self, name):
        """Retreive a specifi unit from the fleet cluster by name

        Args:
            name (str): If specified, only this unit name is returned

        Returns:
            Unit: The unit identified by ``name`` in the fleet cluster

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """
        return Unit(client=self, data=self._single_request('Units.Get', unitName=name))