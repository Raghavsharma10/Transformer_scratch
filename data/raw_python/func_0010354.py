def stop(self):
        """Stops all session activity.

        Blocks until io and writer thread dies
        """
        if self._io_thread is not None:
            self.log.info("Waiting for I/O thread to stop...")
            self.closed = True
            self._io_thread.join()

        if self._writer_thread is not None:
            self.log.info("Waiting for Writer Thread to stop...")
            self.closed = True
            self._writer_thread.join()

        self.log.info("All worker threads stopped.")