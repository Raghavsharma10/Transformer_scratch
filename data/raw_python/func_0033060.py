def _linreg_future(self, series, since, days=20):
        '''
        Predicts future using linear regression.

        :param series:
            A series in which the values will be places.
            The index will not be touched.
            Only the values on dates > `since` will be predicted.
        :param since:
            The starting date from which the future will be predicted.
        :param days:
            Specifies how many past days should be used in the linear
            regression.
        '''
        last_days = pd.date_range(end=since, periods=days)
        hist = self.history(last_days)

        xi = np.array(map(dt2ts, hist.index))
        A = np.array([xi, np.ones(len(hist))])
        y = hist.values
        w = np.linalg.lstsq(A.T, y)[0]

        for d in series.index[series.index > since]:
            series[d] = w[0] * dt2ts(d) + w[1]
            series[d] = 0 if series[d] < 0 else series[d]

        return series