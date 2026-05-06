def set_unit_desired_state(self, unit, desired_state):
        """Update the desired state of a unit running in the cluster

        Args:
            unit (str, Unit): The Unit, or name of the unit to update

            desired_state: State the user wishes the Unit to be in
                          ("inactive", "loaded", or "launched")
        Returns:
            Unit: The unit that was updated

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400
            ValueError: An invalid value was provided for ``desired_state``

        """

        if desired_state not in self._STATES:
            raise ValueError('state must be one of: {0}'.format(
                self._STATES
            ))

        # if we are given an object, grab it's name property
        # otherwise, convert to unicode
        if isinstance(unit, Unit):
            unit = unit.name
        else:
            unit = str(unit)

        self._single_request('Units.Set', unitName=unit, body={
            'desiredState': desired_state
        })

        return self.get_unit(unit)