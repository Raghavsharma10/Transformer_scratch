def stop(self):
        """ Stops listening for events. """

        if not self._is_running:
            return

        pushcenter_logger.debug("[NURESTPushCenter] Stopping...")

        self._thread.stop()
        self._thread.join()

        self._is_running = False
        self._current_connection = None
        self._start_time = None
        self._timeout = None