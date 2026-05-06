async def _poll(self) -> None:
        """Coroutine to poll status of running subprocesses."""
        while True:
            await asyncio.sleep(self._poll_delta)
            for subproc in list(self._running_set):
                subproc._poll()