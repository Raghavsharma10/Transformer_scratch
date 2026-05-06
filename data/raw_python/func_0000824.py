def set_window_property(self, window, name, value):
        """
        Change a window property.

        Example properties you can change are WM_NAME, WM_ICON_NAME, etc.

        :param wid: The window to change a property of.
        :param name: the string name of the property.
        :param value: the string value of the property.
        """
        _libxdo.xdo_set_window_property(self._xdo, window, name, value)