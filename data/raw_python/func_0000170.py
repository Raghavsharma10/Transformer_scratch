async def _start_payloads(self):
        """Start all queued payloads"""
        with self._lock:
            for coroutine in self._payloads:
                task = self.event_loop.create_task(coroutine())
                self._tasks.add(task)
            self._payloads.clear()
        await asyncio.sleep(0)