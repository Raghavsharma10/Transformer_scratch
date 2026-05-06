async def _cancel_payloads(self):
        """Cancel all remaining payloads"""
        for task in self._tasks:
            task.cancel()
            await asyncio.sleep(0)
        for task in self._tasks:
            while not task.done():
                await asyncio.sleep(0.1)
                task.cancel()