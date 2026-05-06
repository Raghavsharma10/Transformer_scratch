def cmd_log(self, reopen=False, rotate=False):
        """Allows managing of uWSGI log related stuff

        :param bool reopen: Reopen log file. Could be required after third party rotation.
        :param bool rotate: Trigger built-in log rotation.

        """
        cmd = b''

        if reopen:
            cmd += b'l'

        if rotate:
            cmd += b'L'

        return self.send_command(cmd)