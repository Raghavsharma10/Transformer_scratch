def _viewbox_set(self, viewbox):
        """ Friend method of viewbox to register itself.
        """
        self._viewbox = viewbox
        # Connect
        viewbox.events.mouse_press.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_release.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_move.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_wheel.connect(self.viewbox_mouse_event)
        viewbox.events.resize.connect(self.viewbox_resize_event)