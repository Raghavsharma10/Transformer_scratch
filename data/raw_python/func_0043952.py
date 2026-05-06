def write(self, data, report_id=b'\0'):
        """ Write an Output report to a HID device.

        This will send the data on the first OUT endpoint, if one exists. If it
        does not, it will be sent the data through the Control Endpoint
        (Endpoint 0).

        :param data:        The data to be sent
        :type data:         str/bytes
        :param report_id:   The Report ID to write to (default: 0x0)

        """
        self._check_device_status()
        bufp = ffi.new("unsigned char[]", len(data)+1)
        buf = ffi.buffer(bufp, len(data)+1)
        buf[0] = report_id
        buf[1:] = data
        rv = hidapi.hid_write(self._device, bufp, len(data)+1)
        if rv == -1:
            raise IOError("Failed to write to HID device.")