def get_device_by_id(self, device_id):
        """Search the list of connected devices by ID.

        device_id param is the integer ID of the device
        """

        # Find the device for the vera device name we are interested in
        found_device = None
        for device in self.get_devices():
            if device.device_id == device_id:
              found_device = device
              # found the first (and should be only) one so we will finish
              break

        if found_device is None:
            logger.debug('Did not find device with {}'.format(device_id))

        return found_device