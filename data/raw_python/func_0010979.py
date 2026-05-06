def stop_stream(self):
        """
        Stops the current stream. Blocks until this is done.
        """

        if self.stream is not None:
            # There is a streaming thread

            logger.warning("Stopping twitter stream...")
            self.stream.disconnect()

            self.stream = None

            # wait a few seconds to allow the streaming to actually stop
            sleep(self.STOP_TIMEOUT)