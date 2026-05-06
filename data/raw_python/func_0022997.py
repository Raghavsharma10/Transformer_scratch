def on_mouse_wheel(self, event):
        """Mouse wheel handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        self.zoom(np.exp(event.delta * (0.01, -0.01)), event.pos)