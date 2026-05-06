def _shutdown(self, message=None, code=0):
        """Shutdown and cleanup everything."""
        if self._shutdown_complete:
            # Make sure we don't accidentally re-run the all cleanup
            sys.exit(code)

        if self.shutdown_callback is not None:
            # Call the shutdown callback with a message suitable for
            # logging and the exit code
            self.shutdown_callback(message, code)

        if self.pidfile is not None:
            self._close_pidfile()

        self._shutdown_complete = True
        sys.exit(code)