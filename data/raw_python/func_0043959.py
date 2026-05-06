def get_indexed_string(self, idx):
        """ Get a string from the device, based on its string index.

        :param idx: The index of the string to get
        :type idx:  int
        :return:    The string at the index
        :rtype:     unicode

        """
        self._check_device_status()
        bufp = ffi.new("wchar_t*")
        rv = hidapi.hid_get_indexed_string(self._device, idx, bufp, 65536)
        if rv == -1:
            raise IOError("Failed to read string with index {0} from HID "
                          "device: {0}"
                          .format(idx, self._get_last_error_string()))
        return ffi.buffer(bufp, 65536)[:].strip()