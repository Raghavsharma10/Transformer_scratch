def set_desired_state(self, state):
        """Update the desired state of a unit.

        Args:
            state (str): The desired state for the unit, must be one of ``_STATES``

        Returns:
            str: The updated state

         Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400
            ValueError: An invalid value for ``state`` was provided
        """
        if state not in self._STATES:
            raise ValueError(
                'state must be one of: {0}'.format(
                    self._STATES
                ))

        # update our internal structure
        self._data['desiredState'] = state

        # if we have a name, then we came from the server
        # and we have a handle to an active client
        # Then update our selves on the server
        if self._is_live():
            self._update('_data', self._client.set_unit_desired_state(self.name, self.desiredState))

        # Return the state
        return self._data['desiredState']