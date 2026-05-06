def stop(self, *args, **kwargs):
        """
        exit
        """
        _logger.debug("Bundle [%s] has been shutting down" %
                      self.bundle.profile["name"])

        if hasattr(self, 'before_stop') and \
           hasattr(self.before_stop, '__call__'):
            _logger.debug("Invoking before_stop...")
            self.before_stop()

        self._conn.disconnect()
        self._session.stop()
        self.stop_event.set()

        # TODO: shutdown all threads
        for thread, stop in self.thread_list:
            stop()
        for thread, stop in self.thread_list:
            thread.join()
        self.is_ready.clear()