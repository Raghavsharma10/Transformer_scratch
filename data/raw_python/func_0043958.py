def get_feature_report(self, report_id, length):
        """ Get a feature report from the device.

        :param report_id:   The Report ID of the report to be read
        :type report_id:    int
        :return:            The report data
        :rtype:             str/bytes

        """
        self._check_device_status()
        bufp = ffi.new("unsigned char[]", length+1)
        buf = ffi.buffer(bufp, length+1)
        buf[0] = report_id
        rv = hidapi.hid_get_feature_report(self._device, bufp, length+1)
        if rv == -1:
            raise IOError("Failed to get feature report from HID device: {0}"
                          .format(self._get_last_error_string()))
        return buf[1:]