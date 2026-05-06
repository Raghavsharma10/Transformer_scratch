def add_to_group(self, devices):
        """Add device(s) to the group."""
        ids = {d.id for d in self.devices_in_group()}
        ids.update(self._device_ids(devices))
        self._set_group(ids)