def read_pid(self):
        """
        Return the PID of the process owning the lock.

        Returns ``None`` if no lock is present.
        """
        try:
            with open(self._path, 'r') as f:
                s = f.read().strip()
                if not s:
                    return None
                return int(s)
        except IOError as e:
            if e.errno == errno.ENOENT:
                return None
            raise