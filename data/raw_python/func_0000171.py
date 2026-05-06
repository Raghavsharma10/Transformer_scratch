async def _reap_payloads(self):
        """Clean up all finished payloads"""
        for task in self._tasks.copy():
            if task.done():
                self._tasks.remove(task)
                if task.exception() is not None:
                    raise task.exception()
        await asyncio.sleep(0)