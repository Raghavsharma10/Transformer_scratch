def destroy_unit(self, unit):
        """Delete a unit from the cluster

        Args:
            unit (str, Unit): The Unit, or name of the unit to delete

        Returns:
            True: The unit was deleted

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """

        # if we are given an object, grab it's name property
        # otherwise, convert to unicode
        if isinstance(unit, Unit):
            unit = unit.name
        else:
            unit = str(unit)

        self._single_request('Units.Delete', unitName=unit)
        return True