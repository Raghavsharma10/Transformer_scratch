def get_mouse_location2(self):
        """
        Get all mouse location-related data.

        :return: a namedtuple with ``x``, ``y``, ``screen_num``
            and ``window`` fields
        """
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        screen_num_ret = ctypes.c_ulong(0)
        window_ret = ctypes.c_ulong(0)
        _libxdo.xdo_get_mouse_location2(
            self._xdo, ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(screen_num_ret), ctypes.byref(window_ret))
        return mouse_location2(x.value, y.value, screen_num_ret.value,
                               window_ret.value)