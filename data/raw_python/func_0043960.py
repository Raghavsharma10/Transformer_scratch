def close(self):
        """ Close connection to HID device.

        Automatically run when a Device object is garbage-collected, though
        manual invocation is recommended.
        """
        self._check_device_status()
        hidapi.hid_close(self._device)
        self._device = None