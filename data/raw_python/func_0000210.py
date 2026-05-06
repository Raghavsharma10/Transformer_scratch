def _reap_payloads(self):
        """Clean up all finished payloads"""
        for thread in self._threads.copy():
            # CapturingThread.join will throw
            if thread.join(timeout=0):
                self._threads.remove(thread)
                self._logger.debug('reaped thread %s', thread)