def _start_payloads(self):
        """Start all queued payloads"""
        with self._lock:
            payloads = self._payloads.copy()
            self._payloads.clear()
        for subroutine in payloads:
            thread = CapturingThread(target=subroutine)
            thread.start()
            self._threads.add(thread)
            self._logger.debug('booted thread %s', thread)
        time.sleep(0)