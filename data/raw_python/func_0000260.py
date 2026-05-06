def stop(self):
        """Stop execution of all current and future payloads"""
        if not self.running.wait(0.2):
            return
        self._logger.debug('runner disabled: %s', self)
        with self._lock:
            self.running.clear()
        self._stopped.wait()