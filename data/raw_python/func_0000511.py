async def _start_payloads(self, nursery):
        """Start all queued payloads"""
        with self._lock:
            for coroutine in self._payloads:
                nursery.start_soon(coroutine)
            self._payloads.clear()
        await trio.sleep(0)