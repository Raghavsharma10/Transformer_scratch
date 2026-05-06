def update_stream(self):
        """
        Restarts the stream with the current list of tracking terms.
        """

        need_to_restart = False

        # If we think we are running, but something has gone wrong in the streaming thread
        # Restart it.
        if self.stream is not None and not self.stream.running:
            logger.warning("Stream exists but isn't running")
            self.listener.error = False
            self.listener.streaming_exception = None
            need_to_restart = True

        # Check if the tracking list has changed
        if self.term_checker.check():
            logger.info("Terms have changed")
            need_to_restart = True

        # If we aren't running and we are allowing unfiltered streams
        if self.stream is None and self.unfiltered:
            need_to_restart = True

        if not need_to_restart:
            return

        logger.info("Restarting stream...")
        
        # Stop any old stream
        self.stop_stream()

        # Start a new stream
        self.start_stream()