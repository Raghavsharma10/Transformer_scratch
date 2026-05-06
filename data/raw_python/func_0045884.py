def destroy(self):
        """Remove a unit from the fleet cluster

        Returns:
            True: The unit was removed

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """

        # if this unit didn't come from fleet, we can't destroy it
        if not self._is_live():
            raise RuntimeError('A unit must be submitted to fleet before it can destroyed.')

        return self._client.destroy_unit(self.name)