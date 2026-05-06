def get_serial_number_string(self):
        """ Get the Serial Number String from the HID device.

        :return:    The Serial Number String
        :rtype:     unicode

        """
        self._check_device_status()
        str_p = ffi.new("wchar_t[]", 255)
        rv = hidapi.hid_get_serial_number_string(self._device, str_p, 255)
        if rv == -1:
            raise IOError("Failed to read serial number string from HID "
                          "device: {0}".format(self._get_last_error_string()))
        return ffi.string(str_p)