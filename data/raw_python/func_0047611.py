def start(self, **kwargs):
        """
        Present the PTY of the container inside the current process.

        This will take over the current process' TTY until the container's PTY
        is closed.
        """

        pty_stdin, pty_stdout, pty_stderr = self.sockets()
        pumps = []

        if pty_stdin and self.interactive:
            pumps.append(io.Pump(io.Stream(self.stdin), pty_stdin, wait_for_output=False))

        if pty_stdout:
            pumps.append(io.Pump(pty_stdout, io.Stream(self.stdout), propagate_close=False))

        if pty_stderr:
            pumps.append(io.Pump(pty_stderr, io.Stream(self.stderr), propagate_close=False))

        if not self.container_info()['State']['Running']:
            self.client.start(self.container, **kwargs)

        flags = [p.set_blocking(False) for p in pumps]

        try:
            with WINCHHandler(self):
                self._hijack_tty(pumps)
        finally:
            if flags:
                for (pump, flag) in zip(pumps, flags):
                    io.set_blocking(pump, flag)