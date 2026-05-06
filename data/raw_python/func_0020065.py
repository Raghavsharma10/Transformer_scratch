def X(self, i, j=slice(None, None, None)):
        '''
        Computes the design matrix at the given *PLD* order and the given
        indices. The columns are the *PLD* vectors for the target at the
        corresponding order, computed as the product of the fractional pixel
        flux of all sets of :py:obj:`n` pixels, where :py:obj:`n` is the *PLD*
        order.

        '''

        X1 = self.fpix[j] / self.norm[j].reshape(-1, 1)
        X = np.product(list(multichoose(X1.T, i + 1)), axis=1).T
        if self.X1N is not None:
            return np.hstack([X, self.X1N[j] ** (i + 1)])
        else:
            return X