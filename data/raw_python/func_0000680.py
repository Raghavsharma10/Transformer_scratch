def modify(self, monitor_id, options):
        """Create a monitor using passed configuration."""
        if not self._state:
            raise InvalidState("State was not properly obtained from the app")
        monitors = self.list()  # Get the latest set of monitors
        obj = None
        for monitor in monitors:
            if monitor_id != monitor['monitor_id']:
                continue
            obj = monitor
        if not monitor_id:
            raise MonitorNotFound("No monitor was found with that term.")
        options['action'] = 'MODIFY'
        options.update(obj)
        payload = self._build_payload(obj['term'], options)
        url = self.ALERTS_MODIFY_URL.format(requestX=self._state[3])
        self._log.debug("Modifying alert using: %s" % url)
        params = json.dumps(payload, separators=(',', ':'))
        data = {'params': params}
        response = self._session.post(url, data=data, headers=self.HEADERS)
        if response.status_code != 200:
            raise ActionError("Failed to create monitor: %s"
                              % response.content)
        return self.list()