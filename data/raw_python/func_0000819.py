def get_active_keys_to_keycode_list(self):
        """Get a list of active keys. Uses XQueryKeymap"""

        try:
            _libxdo.xdo_get_active_keys_to_keycode_list
        except AttributeError:
            # Apparently, this was implemented in a later version..
            raise NotImplementedError()

        keys = POINTER(charcodemap_t)
        nkeys = ctypes.c_int(0)
        _libxdo.xdo_get_active_keys_to_keycode_list(
            self._xdo, ctypes.byref(keys), ctypes.byref(nkeys))

        # todo: make sure this returns a list of charcodemap_t!
        return keys.value