def show(self, visible=True, run=False):
        """Show or hide the canvas

        Parameters
        ----------
        visible : bool
            Make the canvas visible.
        run : bool
            Run the backend event loop.
        """
        self._backend._vispy_set_visible(visible)
        if run:
            self.app.run()