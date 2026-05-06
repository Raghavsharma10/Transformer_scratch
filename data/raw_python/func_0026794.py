def devices(self):
        """Return all known devices.

        :return: list of :class:`Device` or :class:`DeviceGroup` instances.
        """
        devices = []
        count = self.lib.tdGetNumberOfDevices()
        for i in range(count):
            device = DeviceFactory(self.lib.tdGetDeviceId(i), lib=self.lib)
            devices.append(device)
        return devices