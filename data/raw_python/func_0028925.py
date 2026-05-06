def stop(self):
        """Override to implement shutdown steps."""
        LOGGER.info('Attempting to stop the process')
        self.set_state(self.STATE_STOP_REQUESTED)

        # Call shutdown for classes to add shutdown steps
        self.shutdown()

        # Wait for the current run to finish
        while self.is_running and self.is_waiting_to_stop:
            LOGGER.info('Waiting for the process to finish')
            time.sleep(self.SLEEP_UNIT)

        # Change the state to shutting down
        if not self.is_stopping:
            self.set_state(self.STATE_STOPPING)

        # Call a method that may be overwritten to cleanly shutdown
        self.on_shutdown()

        # Change our state
        self.set_state(self.STATE_STOPPED)