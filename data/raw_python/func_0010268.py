def get_device_json(self, use_cached=True):
        """Get the JSON metadata for this device as a python data structure

        If ``use_cached`` is not True, then a web services request will be made
        synchronously in order to get the latest device metatdata.  This will
        update the cached data for this device.

        """
        if not use_cached:
            devicecore_data = self._conn.get_json(
                "/ws/DeviceCore/{}".format(self.get_device_id()))
            self._device_json = devicecore_data["items"][0]  # should only be 1
        return self._device_json