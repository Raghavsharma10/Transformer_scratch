def delete(self, monitor_id):
        """Delete a monitor by ID."""
        if not self._state:
            raise InvalidState("State was not properly obtained from the app")
        monitors = self.list()  # Get the latest set of monitors
        bit = None
        for monitor in monitors:
            if monitor_id != monitor['monitor_id']:
                continue
            bit = monitor['monitor_id']
        if not bit:
            raise MonitorNotFound("No monitor was found with that term.")
        url = self.ALERTS_DELETE_URL.format(requestX=self._state[3])
        self._log.debug("Deleting alert using: %s" % url)
        payload = [None, monitor_id]
        params = json.dumps(payload, separators=(',', ':'))
        data = {'params': params}
        response = self._session.post(url, data=data, headers=self.HEADERS)
        if response.status_code != 200:
            raise ActionError("Failed to delete by ID: %s"
                              % response.content)
        return True