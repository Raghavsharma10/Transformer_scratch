def update(self, counter, f, x_orig, gradient_orig):
        """Perform an update of the linear transformation

           Arguments:
            | ``counter``  --  the iteration counter of the minimizer
            | ``f``  --  the function value at ``x_orig``
            | ``x_orig``  --  the unknowns in original coordinates
            | ``gradient_orig``  --  the gradient in original coordinates

           Return value:
            | ``done_update``  --  True when an update has been done

           The minimizer must reset the search direction method when an updated
           has been done.
        """
        do_update = Preconditioner.update(self, counter, f, x_orig, gradient_orig)
        if do_update:
            # determine a new preconditioner
            N = len(x_orig)
            if self.scales is None:
                self.scales = np.ones(N, float)
            for i in range(N):
                epsilon = self.epsilon/self.scales[i]
                xh = x_orig.copy()
                xh[i] += 0.5*epsilon
                fh = self.fun(xh)
                xl = x_orig.copy()
                xl[i] -= 0.5*epsilon
                fl = self.fun(xl)
                curv = (fh+fl-2*f)/epsilon**2
                self.scales[i] = np.sqrt(abs(curv))
            if self.scales.max() <= 0:
                self.scales = np.ones(N, float)
            else:
                self.scales /= self.scales.max()
                self.scales[self.scales<self.scale_limit] = self.scale_limit
        return do_update