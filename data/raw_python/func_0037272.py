def on_open(self, func):
        """ Function to set the callback for the opening of a stream.

            Can be called manually::

                def open_callback(data):
                    setup_stream()
                client.on_open(open_callback)

            or as a decorator::

                @client.on_open
                def open_callback():
                    setup_stream()
        """
        self._on_open = func
        if self.opened:  # pragma: no cover
            self._on_open(self)
        return func