def add_group(self, name, devices):
        """Add a new device group.

        :return: a :class:`DeviceGroup` instance.
        """
        device = self.add_device(name, "group")
        device.add_to_group(devices)
        return device