def set_file_params(
            self, reopen_on_reload=None, trucate_on_statup=None, max_size=None, rotation_fname=None,
            touch_reopen=None, touch_rotate=None, owner=None, mode=None):
        """Set various parameters related to file logging.

        :param bool reopen_on_reload: Reopen log after reload.

        :param bool trucate_on_statup: Truncate log on startup.

        :param int max_size: Set maximum logfile size in bytes after which log should be rotated.

        :param str|unicode rotation_fname: Set log file name after rotation.

        :param str|unicode|list touch_reopen: Trigger log reopen if the specified file
            is modified/touched.

            .. note:: This can be set to a file touched by ``postrotate`` script of ``logrotate``
                to implement rotation.

        :param str|unicode|list touch_rotate: Trigger log rotation if the specified file
            is modified/touched.

        :param str|unicode owner: Set owner chown() for logs.
        
        :param str|unicode mode: Set mode chmod() for logs.

        """
        self._set('log-reopen', reopen_on_reload, cast=bool)
        self._set('log-truncate', trucate_on_statup, cast=bool)
        self._set('log-maxsize', max_size)
        self._set('log-backupname', rotation_fname)

        self._set('touch-logreopen', touch_reopen, multi=True)
        self._set('touch-logrotate', touch_rotate, multi=True)

        self._set('logfile-chown', owner)
        self._set('logfile-chmod', mode)

        return self._section