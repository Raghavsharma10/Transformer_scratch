def _serve_process(self, slaveFd, serverPid):
        """
            Serves a process by connecting its outputs/inputs to the pty
            slaveFd.  serverPid is the process controlling the master fd
            that passes that output over the socket.
        """
        self.serverPid = serverPid
        if sys.stdin.isatty():
            self.oldTermios = termios.tcgetattr(sys.stdin.fileno())
        else:
            self.oldTermios = None
        self.oldStderr = SavedFile(2, sys, 'stderr')
        self.oldStdout = SavedFile(1, sys, 'stdout')
        self.oldStdin = SavedFile(0, sys, 'stdin')
        self.oldStderr.save(slaveFd, mode="w")
        self.oldStdout.save(slaveFd, mode="w")
        self.oldStdin.save(slaveFd, mode="r")
        os.close(slaveFd)
        self.closed = False