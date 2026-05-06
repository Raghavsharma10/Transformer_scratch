def on_resize(self, event):
        """Resize handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if self._aspect is None:
            return
        w, h = self._canvas.size
        aspect = self._aspect / (w / h)
        self.scale = (self.scale[0], self.scale[0] / aspect)
        self.shader_map()