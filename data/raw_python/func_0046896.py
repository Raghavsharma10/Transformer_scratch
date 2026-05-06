def create_unit(self, name, unit):
        """Create a new Unit in the cluster

        Create and modify Unit entities to communicate to fleet the desired state of the cluster.
        This simply declares what should be happening; the backend system still has to react to
        the changes in this desired state. The actual state of the system is communicated with
        UnitState entities.


        Args:
            name (str): The name of the unit to create
            unit (Unit): The unit to submit to fleet

        Returns:
            Unit: The unit that was created

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """

        self._single_request('Units.Set', unitName=name, body={
            'desiredState': unit.desiredState,
            'options': unit.options
        })

        return self.get_unit(name)