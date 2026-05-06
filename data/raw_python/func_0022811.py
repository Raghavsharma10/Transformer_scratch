def viewbox_mouse_event(self, event):
        """ The ViewBox received a mouse event; update transform
        accordingly.
        Default implementation adjusts scale factor when scolling.

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        BaseCamera.viewbox_mouse_event(self, event)
        if event.type == 'mouse_wheel':
            s = 1.1 ** - event.delta[1]
            self._scale_factor *= s
            if self._distance is not None:
                self._distance *= s
            self.view_changed()