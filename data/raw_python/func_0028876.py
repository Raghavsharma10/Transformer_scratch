def gid(self):
        """Return the group id that the daemon will run with

        :rtype: int

        """
        if not self._gid:
            if self.controller.config.daemon.group:
                self._gid = grp.getgrnam(self.config.daemon.group).gr_gid
            else:
                self._gid = os.getgid()
        return self._gid