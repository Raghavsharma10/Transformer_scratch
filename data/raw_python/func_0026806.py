def devices_in_group(self):
        """Fetch list of devices in group."""
        try:
            devices = self.get_parameter('devices')
        except AttributeError:
            return []

        ctor = DeviceFactory
        return [ctor(int(x), lib=self.lib) for x in devices.split(',') if x]