def select_by_command_record(self, crec):
        """
        Yield records that matches to `crec`.

        All attributes of `crec` except for `environ` are concerned.

        """
        keys = ['command_history_id', 'command', 'session_history_id',
                'cwd', 'terminal',
                'start', 'stop', 'exit_code']
        sql = """
        SELECT
            command_history.id, CL.command, session_id,
            DL.directory, TL.terminal,
            start_time, stop_time, exit_code
        FROM command_history
        LEFT JOIN command_list AS CL ON command_id = CL.id
        LEFT JOIN directory_list AS DL ON directory_id = DL.id
        LEFT JOIN terminal_list AS TL ON terminal_id = TL.id
        WHERE
            (CL.command   = ? OR (CL.command   IS NULL AND ? IS NULL)) AND
            (DL.directory = ? OR (DL.directory IS NULL AND ? IS NULL)) AND
            (TL.terminal  = ? OR (TL.terminal  IS NULL AND ? IS NULL)) AND
            (start_time   = ? OR (start_time   IS NULL AND ? IS NULL)) AND
            (stop_time    = ? OR (stop_time    IS NULL AND ? IS NULL)) AND
            (exit_code    = ? OR (exit_code    IS NULL AND ? IS NULL))
        """
        desired_row = [
            crec.command, normalize_directory(crec.cwd), crec.terminal,
            convert_ts(crec.start), convert_ts(crec.stop), crec.exit_code]
        params = list(itertools.chain(*zip(desired_row, desired_row)))
        return self._select_rows(CommandRecord, keys, sql, params)