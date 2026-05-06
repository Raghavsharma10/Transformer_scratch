def set_emperor_command_params(
            self, command_socket=None,
            wait_for_command=None, wait_for_command_exclude=None):
        """Emperor commands related parameters.

        * http://uwsgi-docs.readthedocs.io/en/latest/tutorials/EmperorSubscriptions.html

        :param str|unicode command_socket: Enable the Emperor command socket.
            It is a channel allowing external process to govern vassals.

        :param bool wait_for_command: Always wait for a 'spawn' Emperor command before starting a vassal.

        :param str|unicode|list[str|unicode] wait_for_command_exclude: Vassals that will ignore ``wait_for_command``.

        """
        self._set('emperor-command-socket', command_socket)
        self._set('emperor-wait-for-command', wait_for_command, cast=bool)
        self._set('emperor-wait-for-command-ignore', wait_for_command_exclude, multi=True)

        return self._section