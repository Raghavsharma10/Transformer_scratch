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
        if Preconditioner.update(self, counter, f, x_orig, gradient_orig):
            # determine a new preconditioner
            hessian = compute_fd_hessian(self.fun, x_orig, self.epsilon)
            evals, evecs = np.linalg.eigh(hessian)
            self.scales = np.sqrt(abs(evals))+self.epsilon
            self.rotation = evecs
            return True
        return False