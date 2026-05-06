def start_stream_subscriber(self):
        """ Starts the stream consumer's main loop.

            Called when the stream consumer has been set up with the correct callbacks.
        """
        if not self._stream_process_started:  # pragma: no cover
            if sys.platform.startswith("win"):  # if we're on windows we can't expect multiprocessing to work
                self._stream_process_started = True
                self._stream()
            self._stream_process_started = True
            self._stream_process.start()