def _poll(self) -> None:
        """Check the status of the wrapped running subprocess.

        Note:
            This should only be called on currently-running tasks.

        """
        if self._subprocess is None:
            raise SublemonLifetimeError(
                'Attempted to poll a non-active subprocess')
        elif self._subprocess.returncode is not None:
            self._exit_code = self._subprocess.returncode
            self._done_running_evt.set()
            self._server._running_set.remove(self)
            self._server._sem.release()