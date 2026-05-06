def list_unit_states(self, machine_id=None, unit_name=None):
        """Return the current UnitState for the fleet cluster

        Args:
            machine_id (str): filter all UnitState objects to those
                              originating from a specific machine

            unit_name (str):  filter all UnitState objects to those related
                              to a specific unit

        Yields:
            UnitState: The next UnitState in the cluster

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """
        for page in self._request('UnitState.List', machineID=machine_id, unitName=unit_name):
            for state in page.get('states', []):
                yield UnitState(data=state)