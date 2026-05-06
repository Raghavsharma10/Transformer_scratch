def getSlaveStatus(self):
        """Returns status of replication slaves.
        
        @return: Dictionary of status items.
        
        """
        info_dict = {}
        if self.checkVersion('9.1'):
            cols = ['procpid', 'usename', 'application_name', 
                    'client_addr', 'client_port', 'backend_start', 'state', 
                    'sent_location', 'write_location', 'flush_location', 
                    'replay_location', 'sync_priority', 'sync_state',]
            cur = self._conn.cursor()
            cur.execute("""SELECT %s FROM pg_stat_replication;""" 
                        % ','.join(cols))
            rows = cur.fetchall()
            for row in rows:
                info_dict[row[0]] = dict(zip(cols[1:], row[1:]))
        else:
            return None
        return info_dict