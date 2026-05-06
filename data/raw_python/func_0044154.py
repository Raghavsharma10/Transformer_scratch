async def stdout(self) -> AsyncGenerator[str, None]:
        """Asynchronous generator for lines from subprocess stdout."""
        await self.wait_running()
        async for line in self._subprocess.stdout:  # type: ignore
            yield line