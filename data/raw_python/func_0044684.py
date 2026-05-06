async def start(self) -> None:
        """Coroutine to run this server."""
        if self._is_running:
            raise SublemonRuntimeError(
                'Attempted to start an already-running `Sublemon` instance')

        self._poll_task = asyncio.ensure_future(self._poll())
        self._is_running = True