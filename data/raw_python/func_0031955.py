def get_full_command_record(self, command_history_id,
                                merge_session_environ=True):
        """
        Get fully retrieved :class:`CommandRecord` instance by ID.

        By "fully", it means that complex slots such as `environ` and
        `pipestatus` are available.

        :type    command_history_id: int
        :type merge_session_environ: bool

        """
        with self.connection() as db:
            crec = self._select_command_record(db, command_history_id)
            crec.pipestatus = self._get_pipestatus(db, command_history_id)
            # Set environment variables
            cenv = self._select_environ(db, 'command', command_history_id)
            crec.environ.update(cenv)
            if merge_session_environ:
                senv = self._select_environ(
                    db, 'session', crec.session_history_id)
                crec.environ.update(senv)
        return crec