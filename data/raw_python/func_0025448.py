def gerrity_score(self):
        """
        Gerrity Score, which weights each cell in the contingency table by its observed relative frequency.
        :return:
        """
        k = self.table.shape[0]
        n = float(self.table.sum())
        p_o = self.table.sum(axis=0) / n
        p_sum = np.cumsum(p_o)[:-1]
        a = (1.0 - p_sum) / p_sum
        s = np.zeros(self.table.shape, dtype=float)
        for (i, j) in np.ndindex(*s.shape):
            if i == j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:j]) + np.sum(a[j:k-1]))
            elif i < j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:i]) - (j - i) + np.sum(a[j:k-1]))
            else:
                s[i, j] = s[j, i]
        return np.sum(self.table / float(self.table.sum()) * s)