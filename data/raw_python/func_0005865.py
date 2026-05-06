def log_into(self, target, before_priv_drop=True):
        """Simple file or UDP logging.

        .. note:: This doesn't require any Logger plugin and can be used
            if no log routing is required.

        :param str|unicode target: Filepath or UDP address.

        :param bool before_priv_drop: Whether to log data before or after privileges drop.

        """
        command = 'logto'

        if not before_priv_drop:
            command += '2'

        self._set(command, target)

        return self._section