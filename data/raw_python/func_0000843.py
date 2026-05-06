def get_active_modifiers(self):
        """
        Get a list of active keys. Uses XQueryKeymap.

        :return: list of charcodemap_t instances
        """
        keys = ctypes.pointer(charcodemap_t())
        nkeys = ctypes.c_int(0)

        _libxdo.xdo_get_active_modifiers(
            self._xdo, ctypes.byref(keys), ctypes.byref(nkeys))
        return [keys[i] for i in range(nkeys.value)]