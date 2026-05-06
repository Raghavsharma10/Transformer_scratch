async def stop(self) -> None:
        """Coroutine to stop execution of this server."""
        if not self._is_running:
            raise SublemonRuntimeError(
                'Attempted to stop an already-stopped `Sublemon` instance')

        await self.block()
        self._poll_task.cancel()
        self._is_running = False
        with suppress(asyncio.CancelledError):
            await self._poll_task