def spawn(self, *cmds: str) -> List[SublemonSubprocess]:
        """Coroutine to spawn shell commands.

        If `max_concurrency` is reached during the attempt to spawn the
        specified subprocesses, excess subprocesses will block while attempting
        to acquire this server's semaphore.

        """
        if not self._is_running:
            raise SublemonRuntimeError(
                'Attempted to spawn subprocesses from a non-started server')

        subprocs = [SublemonSubprocess(self, cmd) for cmd in cmds]
        for sp in subprocs:
            asyncio.ensure_future(sp.spawn())
        return subprocs