def w_func(self, X, d, n):
        """Evaluate the (possibly recursive) warping function and its derivatives.
        
        Parameters
        ----------
        X : array, (`M`,)
            The points (from dimension `d`) to evaluate the warping function at.
        d : int
            The dimension to warp.
        n : int
            The derivative order to compute. So far only 0 and 1 are supported.
        """
        if n == 0:
            wX = self.w(X, d, 0)
            if isinstance(self.k, WarpedKernel):
                wX = self.k.w_func(wX, d, 0)
            return wX
        elif n == 1:
            wXn = self.w(X, d, n)
            if isinstance(self.k, WarpedKernel):
                wX = self.w_func(X, d, 0)
                wXn *= self.k.w_func(wX, d, n)
            return wXn
        else:
            raise ValueError("Derivative orders greater than one are not supported!")