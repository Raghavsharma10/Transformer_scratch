async def block(self) -> None:
        """Block until all running and pending subprocesses have finished."""
        await asyncio.gather(
            *itertools.chain(
                (sp.wait_done() for sp in self._running_set),
                (sp.wait_done() for sp in self._pending_set)))