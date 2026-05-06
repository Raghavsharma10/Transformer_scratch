def isfinite(self):
        "Test whether the predicted values are finite"
        if self._multiple_outputs:
            if self.hy_test is not None:
                r = [(hy.isfinite() and (hyt is None or hyt.isfinite()))
                     for hy, hyt in zip(self.hy, self.hy_test)]
            else:
                r = [hy.isfinite() for hy in self.hy]
            return np.all(r)
        return self.hy.isfinite() and (self.hy_test is None or
                                       self.hy_test.isfinite())