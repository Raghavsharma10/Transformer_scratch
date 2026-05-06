async def spawn(self):
        """Spawn the command wrapped in this object as a subprocess."""
        self._server._pending_set.add(self)
        await self._server._sem.acquire()
        self._subprocess = await asyncio.create_subprocess_shell(
            self._cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self._began_at = datetime.now()
        if self in self._server._pending_set:
            self._server._pending_set.remove(self)
        self._server._running_set.add(self)
        self._began_running_evt.set()