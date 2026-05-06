def _maybe_classify(self, y, k, cutoffs):
        '''Helper method for classifying continuous data.

        '''

        rows, cols = y.shape
        if cutoffs is None:
            if self.fixed:
                mcyb = mc.Quantiles(y.flatten(), k=k)
                yb = mcyb.yb.reshape(y.shape)
                cutoffs = mcyb.bins
                k = len(cutoffs)
                return yb, cutoffs[:-1], k
            else:
                yb = np.array([mc.Quantiles(y[:, i], k=k).yb for i in
                               np.arange(cols)]).transpose()
                return yb, None, k
        else:
            cutoffs = list(cutoffs) + [np.inf]
            cutoffs = np.array(cutoffs)
            yb = mc.User_Defined(y.flatten(), np.array(cutoffs)).yb.reshape(
                y.shape)
            k = len(cutoffs)
            return yb, cutoffs[:-1], k