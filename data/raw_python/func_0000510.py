async def _await_all(self):
        """Async component of _run"""
        delay = 0.0
        # we run a top-level nursery that automatically reaps/cancels for us
        async with trio.open_nursery() as nursery:
            while self.running.is_set():
                await self._start_payloads(nursery=nursery)
                await trio.sleep(delay)
                delay = min(delay + 0.1, 1.0)
            # cancel the scope to cancel all payloads
            nursery.cancel_scope.cancel()