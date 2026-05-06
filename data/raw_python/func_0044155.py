async def stderr(self) -> AsyncGenerator[str, None]:
        """Asynchronous generator for lines from subprocess stderr."""
        await self.wait_running()
        async for line in self._subprocess.stderr:  # type: ignore
            yield line