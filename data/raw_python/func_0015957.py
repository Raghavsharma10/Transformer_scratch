def _actuator_on_off(self, on_off, service_location_id, actuator_id,
                         duration=None):
        """
        Turn actuator on or off

        Parameters
        ----------
        on_off : str
            'on' or 'off'
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
        url = urljoin(URLS['servicelocation'], service_location_id,
                      "actuator", actuator_id, on_off)
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        if duration is not None:
            data = {"duration": duration}
        else:
            data = {}
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
        return r