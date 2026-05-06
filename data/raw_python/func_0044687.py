async def iter_lines(
            self,
            *cmds: str,
            stream: str='both') -> AsyncGenerator[str, None]:
        """Coroutine to spawn commands and yield text lines from stdout."""
        sps = self.spawn(*cmds)
        if stream == 'both':
            agen = amerge(
                amerge(*[sp.stdout for sp in sps]),
                amerge(*[sp.stderr for sp in sps]))
        elif stream == 'stdout':
            agen = amerge(*[sp.stdout for sp in sps])
        elif stream == 'stderr':
            agen = amerge(*[sp.stderr for sp in sps])
        else:
            raise SublemonRuntimeError(
                'Invalid `stream` kwarg received: `' + str(stream) + '`')
        async for line in agen:
            yield line.decode('utf-8').rstrip()