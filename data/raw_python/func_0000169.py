async def _run_payloads(self):
        """Async component of _run"""
        delay = 0.0
        try:
            while self.running.is_set():
                await self._start_payloads()
                await self._reap_payloads()
                await asyncio.sleep(delay)
                delay = min(delay + 0.1, 1.0)
        except Exception:
            await self._cancel_payloads()
            raise