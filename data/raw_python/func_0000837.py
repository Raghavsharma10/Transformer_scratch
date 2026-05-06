def get_number_of_desktops(self):
        """
        Get the current number of desktops.
        Uses ``_NET_NUMBER_OF_DESKTOPS`` of the EWMH spec.

        :param ndesktops:
            pointer to long where the current number of desktops is stored
        """
        ndesktops = ctypes.c_long(0)
        _libxdo.xdo_get_number_of_desktops(self._xdo, ctypes.byref(ndesktops))
        return ndesktops.value