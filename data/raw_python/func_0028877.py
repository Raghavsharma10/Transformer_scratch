def uid(self):
        """Return the user id that the process will run as

        :rtype: int

        """
        if not self._uid:
            if self.config.daemon.user:
                self._uid = pwd.getpwnam(self.config.daemon.user).pw_uid
            else:
                self._uid = os.getuid()
        return self._uid