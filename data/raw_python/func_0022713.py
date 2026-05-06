def close(self):
        """Close the canvas

        Notes
        -----
        This will usually destroy the GL context. For Qt, the context
        (and widget) will be destroyed only if the widget is top-level.
        To avoid having the widget destroyed (more like standard Qt
        behavior), consider making the widget a sub-widget.
        """
        if self._backend is not None and not self._closed:
            self._closed = True
            self.events.close()
            self._backend._vispy_close()
        forget_canvas(self)