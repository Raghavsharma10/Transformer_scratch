async def gather(self, *cmds: str) -> Tuple[int]:
        """Coroutine to spawn subprocesses and block until completion.

        Note:
            The same `max_concurrency` restriction that applies to `spawn`
            also applies here.

        Returns:
            The exit codes of the spawned subprocesses, in the order they were
            passed.

        """
        subprocs = self.spawn(*cmds)
        subproc_wait_coros = [subproc.wait_done() for subproc in subprocs]
        return await asyncio.gather(*subproc_wait_coros)