def last_chain(self):
        '''
        Leaves only the last chain for each object.

        Chain is a series of consecutive versions where
        `_end` of one is `_start` of another.
        '''
        cols = self.columns.tolist()
        i_oid = cols.index('_oid')
        i_start = cols.index('_start')
        i_end = cols.index('_end')

        start_map = {}
        end_map = {}
        for row in self.values:
            oid = row[i_oid]
            if oid not in start_map:
                start_map[oid] = set()
                end_map[oid] = set()
            start_map[oid].add(row[i_start])
            end_map[oid].add(row[i_end])

        cutoffs = {}
        for oid in start_map:
            maxend = pd.NaT if pd.NaT in end_map[oid] else max(end_map[oid])
            ends = end_map[oid] - start_map[oid] - set([maxend])
            cutoffs[oid] = None if len(ends) == 0 else max(ends)

        vals = [row for row in self.values
                if cutoffs[row[i_oid]] is None
                or cutoffs[row[i_oid]] < row[i_start]]

        return pd.DataFrame(vals, columns=cols)