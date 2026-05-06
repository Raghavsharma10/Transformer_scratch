def send_feature_report(self, data, report_id=0x0):
        """ Send a Feature report to the device.

        Feature reports are sent over the Control endpoint as a Set_Report
        transfer.

        :param data:        The data to send
        :type data:         str/bytes
        :param report_id:   The Report ID to send to
        :type report_id:    int

        """
        self._check_device_status()
        bufp = ffi.new("unsigned char[]", len(data)+1)
        buf = ffi.buffer(bufp, len(data)+1)
        buf[0] = report_id
        buf[1:] = data
        rv = hidapi.hid_send_feature_report(self._device, bufp, len(bufp))
        if rv == -1:
            raise IOError("Failed to send feature report to HID device: {0}"
                          .format(self._get_last_error_string()))