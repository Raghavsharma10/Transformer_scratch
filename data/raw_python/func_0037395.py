def pid(self):
        """The server's PID (None if not running).
        """
        # We can't possibly be running if our base_pathname isn't defined.
        if not self.base_pathname:
            return None
        try:
            pidfile = os.path.join(self.base_pathname, 'postmaster.pid')
            return int(open(pidfile).readline())
        except (IOError, OSError):
            return None