def set_window_class(self, window, name, class_):
        """
        Change the window's classname and or class.

        :param name: The new class name. If ``None``, no change.
        :param class_: The new class. If ``None``, no change.
        """
        _libxdo.xdo_set_window_class(self._xdo, window, name, class_)