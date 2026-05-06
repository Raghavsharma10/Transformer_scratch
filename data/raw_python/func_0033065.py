def persistent_oid_counts(self, dates):
        '''
        Counts how many objects (identified by their oids) existed before
        or on a given date.

        :param dates: list of the dates the count should be computed.
        '''
        total = pd.Series([self.on_date(d)._oid for d in dates],
                          index=dates)
        for i in range(1, total.size):
            a1 = total[total.index[i - 1]]
            a2 = total[total.index[i]]
            total[total.index[i]] = list(set(a1) | set(a2))
        return total.apply(len)