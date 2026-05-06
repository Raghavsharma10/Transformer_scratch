def wait_for_completion(self, waiting_func=None):
        """This is a blocking function call that will wait for the queuing
        thread to complete.

        parameters:
            waiting_func - this function will be called every one second while
                           waiting for the queuing thread to quit.  This allows
                           for logging timers, status indicators, etc."""
        self.logger.debug("waiting to join queuingThread")
        self._responsive_join(self.queuing_thread, waiting_func)