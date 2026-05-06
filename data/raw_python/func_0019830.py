def getDatabaseStats(self):
        """Returns database block read, transaction and tuple stats for each 
        database.
        
        @return: Nested dictionary of stats.
        
        """
        headers = ('datname', 'numbackends', 'xact_commit', 'xact_rollback', 
                   'blks_read', 'blks_hit', 'tup_returned', 'tup_fetched', 
                   'tup_inserted', 'tup_updated', 'tup_deleted', 'disk_size')
        cur = self._conn.cursor()
        cur.execute("SELECT %s, pg_database_size(datname) FROM pg_stat_database;" 
                    % ",".join(headers[:-1]))
        rows = cur.fetchall()
        dbstats = self._createStatsDict(headers, rows)
        totals = self._createTotalsDict(headers, rows)
        return {'databases': dbstats, 'totals': totals}