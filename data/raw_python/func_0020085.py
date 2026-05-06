def cv_compute(self, b, A, B, C, mK, f, m1, m2):
        '''
        Compute the model (cross-validation step only) for chunk :py:obj:`b`.

        '''

        A = np.sum([l * a for l, a in zip(self.lam[b], A)
                    if l is not None], axis=0)
        B = np.sum([l * b for l, b in zip(self.lam[b], B)
                    if l is not None], axis=0)
        W = np.linalg.solve(mK + A + C, f)
        if self.transit_model is None:
            model = np.dot(B, W)
        else:
            w_pld = np.concatenate([l * np.dot(self.X(n, m2).T, W)
                                    for n, l in enumerate(self.lam[b])
                                    if l is not None])
            model = np.dot(np.hstack(
                [self.X(n, m1) for n, l in enumerate(self.lam[b])
                 if l is not None]), w_pld)
        model -= np.nanmedian(model)

        return model