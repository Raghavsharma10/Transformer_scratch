def getLockStatsMode(self):
        """Returns the number of active lock discriminated by lock mode.
        
        @return: : Dictionary of stats.
        
        """
        info_dict = {'all': dict(zip(self.lockModes, (0,) * len(self.lockModes))),
                     'wait': dict(zip(self.lockModes, (0,) * len(self.lockModes)))}
        cur = self._conn.cursor()
        cur.execute("SELECT TRIM(mode, 'Lock'), granted, COUNT(*) FROM pg_locks "
                    "GROUP BY TRIM(mode, 'Lock'), granted;")
        rows = cur.fetchall()
        for (mode, granted, cnt) in rows:
            info_dict['all'][mode] += cnt
            if not granted:
                info_dict['wait'][mode] += cnt
        return info_dict