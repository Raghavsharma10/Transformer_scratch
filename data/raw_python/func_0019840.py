def getProcessStatus(self):
        """Returns number of processes discriminated by state.
        
        @return: Dictionary mapping process state to number of processes.
        
        """
        info_dict = {}
        cur = self._conn.cursor()
        cur.execute("""SHOW FULL PROCESSLIST;""")
        rows = cur.fetchall()
        if rows:
            for row in rows:
                if row[6] == '':
                    state = 'idle'
                elif row[6] is None:
                    state = 'other'
                else:
                    state = str(row[6]).replace(' ', '_').lower()
                info_dict[state] = info_dict.get(state, 0) + 1
        return info_dict