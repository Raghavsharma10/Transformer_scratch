async def wait_done(self) -> int:
        """Coroutine to wait for subprocess run completion.

        Returns:
            The exit code of the subprocess.

        """
        await self._done_running_evt.wait()
        if self._exit_code is None:
            raise SublemonLifetimeError(
                'Subprocess exited abnormally with `None` exit code')
        return self._exit_code