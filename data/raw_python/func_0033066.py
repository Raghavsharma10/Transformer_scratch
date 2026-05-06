def last_versions_with_age(self, col_name='age'):
        '''
        Leaves only the latest version for each object.
        Adds a new column which represents age.
        The age is computed by subtracting _start of the oldest version
        from one of these possibilities::

            # psuedo-code
            if self._rbound is None:
                if latest_version._end is pd.NaT:
                    current_time is used
                else:
                    min(current_time, latest_version._end) is used
            else:
                if latest_version._end is pd.NaT:
                    self._rbound is used
                else:
                    min(self._rbound, latest_version._end) is used

        :param index: name of the new column.
        '''
        min_start_map = {}
        max_start_map = {}
        max_start_ser_map = {}

        cols = self.columns.tolist()
        i_oid = cols.index('_oid')
        i_start = cols.index('_start')
        i_end = cols.index('_end')
        for row in self.values:
            s = row[i_start]
            oid = row[i_oid]
            mins = min_start_map.get(oid, s)
            if s <= mins:
                min_start_map[oid] = s
            maxs = max_start_map.get(oid, s)
            if s >= maxs:
                max_start_map[oid] = s
                max_start_ser_map[oid] = row

        vals = max_start_ser_map.values()
        cut_ts = datetime.utcnow()
        ages = []
        for row in vals:
            end = row[i_end]
            end = cut_ts if end is pd.NaT else min(cut_ts, end)
            age = end - min_start_map[row[i_oid]]
            age = age - timedelta(microseconds=age.microseconds)
            ages.append(age)

        res = pd.DataFrame(max_start_ser_map.values(), columns=cols)
        res[col_name] = pd.Series(ages, index=res.index)
        return res