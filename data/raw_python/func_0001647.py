def check_output(self, cmd):
        """Calls a command through SSH and returns its output.
        """
        ret, output = self._call(cmd, True)
        if ret != 0:  # pragma: no cover
            raise RemoteCommandFailure(command=cmd, ret=ret)
        logger.debug("Output: %r", output)
        return output