def one_version(self, index=0):
        '''
        Leaves only one version for each object.

        :param index: List-like index of the version.  0 == first; -1 == last
        '''
        def prep(df):
            start = sorted(df._start.tolist())[index]
            return df[df._start == start]

        return pd.concat([prep(df) for _, df in self.groupby(self._oid)])