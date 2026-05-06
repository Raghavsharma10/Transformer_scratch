def create_native(self):
        """ Create the native widget if not already done so. If the widget
        is already created, this function does nothing.
        """
        if self._backend is not None:
            return
        # Make sure that the app is active
        assert self._app.native
        # Instantiate the backend with the right class
        self._app.backend_module.CanvasBackend(self, **self._backend_kwargs)
        # self._backend = set by BaseCanvasBackend
        self._backend_kwargs = None  # Clean up

        # Connect to draw event (append to the end)
        # Process GLIR commands at each paint event
        self.events.draw.connect(self.context.flush_commands, position='last')
        if self._autoswap:
            self.events.draw.connect((self, 'swap_buffers'),
                                     ref=True, position='last')