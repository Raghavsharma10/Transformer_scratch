def setCmd(self, cmd):
        """Check the cmd is valid, FrameError will be raised if its not."""
        cmd = cmd.upper()
        if cmd not in VALID_COMMANDS:
            raise FrameError("The cmd '%s' is not valid! It must be one of '%s' (STOMP v%s)." % (
                cmd, VALID_COMMANDS, STOMP_VERSION)
            )
        else:
            self._cmd = cmd