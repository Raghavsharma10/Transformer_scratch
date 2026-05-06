def solve(self,b,overwrite_b=False,check_finite=True):
        """
        solve A \ b
        """
        if self._s is not None:
            res = self._U.T.dot(b)
            res /= self._s[:,np.newaxis]
            res = self._U.dot(res)
        elif self._chol is not None:
            res = la.cho_solve((self._chol,self._lower),b=b,overwrite_b=overwrite_b,check_finite=check_finite)
        else:
            res = np.zeros(b.shape)
        return res