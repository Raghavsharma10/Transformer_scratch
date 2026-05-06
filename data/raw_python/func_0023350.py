def on_close(self, event):
        """Close event handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        self.events.mouse_press.disconnect(self._process_mouse_event)
        self.events.mouse_move.disconnect(self._process_mouse_event)
        self.events.mouse_release.disconnect(self._process_mouse_event)
        self.events.mouse_wheel.disconnect(self._process_mouse_event)