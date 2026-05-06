def get_device_by_name(self, device_name):
        """Search the list of connected devices by name.

        device_name param is the string name of the device
        """

        # Find the device for the vera device name we are interested in
        found_device = None
        for device in self.get_devices():
            if device.name == device_name:
              found_device = device
              # found the first (and should be only) one so we will finish
              break

        if found_device is None:
            logger.debug('Did not find device with {}'.format(device_name))

        return found_device