def get_runtime_dir(self, default=True):
        """Directory to store runtime files.
        See ``.replace_placeholders()``

        .. note:: This can be used to store PID files, sockets, master FIFO, etc.

        :param bool default: Whether to return [system] default if not set.

        :rtype: str|unicode
        """
        dir_ = self._runtime_dir

        if not dir_ and default:
            uid = self.main_process.get_owner()[0]
            dir_ = '/run/user/%s/' % uid if uid else '/run/'

        return dir_