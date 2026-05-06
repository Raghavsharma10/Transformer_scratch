def actuator_on(self, service_location_id, actuator_id, duration=None):
        """
        Turn actuator on

        Parameters
        ----------
        service_location_id : int
        actuator_id : int
        duration : int, optional
            300,900,1800 or 3600 , specifying the time in seconds the actuator
            should be turned on. Any other value results in turning on for an
            undetermined period of time.

        Returns
        -------
        requests.Response
        """
        return self._actuator_on_off(
            on_off='on', service_location_id=service_location_id,
            actuator_id=actuator_id, duration=duration)