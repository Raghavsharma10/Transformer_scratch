def run(self):
        """
        Execute all current and future payloads

        Blocks and executes payloads until :py:meth:`stop` is called.
        It is an error for any orphaned payload to return or raise.
        """
        self._logger.info('runner started: %s', self)
        try:
            with self._lock:
                assert not self.running.is_set() and self._stopped.is_set(), 'cannot re-run: %s' % self
                self.running.set()
                self._stopped.clear()
            self._run()
        except Exception:
            self._logger.exception('runner aborted: %s', self)
            raise
        else:
            self._logger.info('runner stopped: %s', self)
        finally:
            with self._lock:
                self.running.clear()
                self._stopped.set()