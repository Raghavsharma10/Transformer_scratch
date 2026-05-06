def getXlogStatus(self):
        """Returns Transaction Logging or Recovery Status.
        
        @return: Dictionary of status items.
        
        """
        inRecovery = None
        if self.checkVersion('9.0'):
            inRecovery = self._simpleQuery("SELECT pg_is_in_recovery();")
        cur = self._conn.cursor()
        if inRecovery:
            cols = ['pg_last_xlog_receive_location()', 
                    'pg_last_xlog_replay_location()',]
            headers = ['xlog_receive_location',
                       'xlog_replay_location',]
            if self.checkVersion('9.1'):
                cols.extend(['pg_last_xact_replay_timestamp()',
                             'pg_is_xlog_replay_paused()',])
                headers.extend(['xact_replay_timestamp', 
                                'xlog_replay_paused',])
            cur.execute("""SELECT %s;""" % ','.join(cols))
            headers = ('xlog_receive_location', 'xlog_replay_location')
        else:
            cur.execute("""SELECT
                pg_current_xlog_location(), 
                pg_xlogfile_name(pg_current_xlog_location());""")
            headers = ('xlog_location', 'xlog_filename')
        row = cur.fetchone()
        info_dict = dict(zip(headers, row))
        if inRecovery is not None:
            info_dict['in_recovery'] = inRecovery
        return info_dict