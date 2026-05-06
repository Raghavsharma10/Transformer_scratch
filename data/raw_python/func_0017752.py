def is_running(self):
        """
        Check if the daemon is running.
        """
        pid = self.get_pid()
        if pid is None:
            return False
        # The PID file may still exist even if the daemon isn't running,
        # for example if it has crashed.
        try:
            os.kill(pid, 0)
        except OSError as e:
            if e.errno == errno.ESRCH:
                # In this case the PID file shouldn't have existed in
                # the first place, so we remove it
                self.pid_file.release()
                return False
            # We may also get an exception if we're not allowed to use
            # kill on the process, but that means that the process does
            # exist, which is all we care about here.
        return True