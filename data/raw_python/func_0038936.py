def safe_date(self, x):
        """Transform x[self.col_name] into a date string.

        Args:
            x(dict like / pandas.Series): Row containing data to cast safely.

        Returns:
            str
        """

        t = x[self.col_name]
        if np.isnan(t):
            return t

        elif np.isposinf(t):
            t = sys.maxsize

        elif np.isneginf(t):
            t = -sys.maxsize

        tmp = time.localtime(float(t) / 1e9)
        return time.strftime(self.date_format, tmp)