def read(self, length, timeout_ms=0, blocking=False):
        """ Read an Input report from a HID device with timeout.

        Input reports are returned to the host through the `INTERRUPT IN`
        endpoint. The first byte will contain the Report number if the device
        uses numbered reports.
        By default reads are non-blocking, i.e. the method will return
        `None` if no data was available. Blocking reads can be enabled with
        :param blocking:. Additionally, a timeout for the read can be
        specified.

        :param length:      The number of bytes to read. For devices with
                            multiple reports, make sure to read an extra byte
                            for the report number.
        :param timeout_ms:  Timeout in miliseconds
        :type timeout_ms:   int
        :param blocking:    Block until data is available

        """
        self._check_device_status()
        bufp = ffi.new("unsigned char[]", length)
        if not timeout_ms and blocking:
            timeout_ms = -1
        if timeout_ms:
            rv = hidapi.hid_read_timeout(self._device, bufp, length,
                                         timeout_ms)
        else:
            rv = hidapi.hid_read(self._device, bufp, length)
        if rv == -1:
            raise IOError("Failed to read from HID device: {0}"
                          .format(self._get_last_error_string()))
        elif rv == 0:
            return None
        else:
            return ffi.buffer(bufp, rv)[:]