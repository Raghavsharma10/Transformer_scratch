def _pid_is_alive(cls, pid, timeout):
        """Check if a PID is alive with a timeout."""
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return False

        try:
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            return True

        return False