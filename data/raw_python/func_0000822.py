def translate_window_with_sizehint(self, window, width, height):
        """
        Apply a window's sizing hints (if any) to a given width and height.

        This function wraps XGetWMNormalHints() and applies any
        resize increment and base size to your given width and height values.

        :param window: the window to use
        :param width: the unit width you want to translate
        :param height: the unit height you want to translate
        :return: (width, height)
        """
        width_ret = ctypes.c_uint(0)
        height_ret = ctypes.c_uint(0)
        _libxdo.xdo_translate_window_with_sizehint(
            self._xdo, window, width, height,
            ctypes.byref(width_ret),
            ctypes.byref(height_ret))
        return width_ret.value, height_ret.value