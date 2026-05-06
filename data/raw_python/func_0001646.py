def check_call(self, cmd):
        """Calls a command through SSH.
        """
        ret, _ = self._call(cmd, False)
        if ret != 0:  # pragma: no cover
            raise RemoteCommandFailure(command=cmd, ret=ret)