def update(self, counter, f, x_orig, gradient_orig):
        """Perform an update of the linear transformation

           Arguments:
            | ``counter``  --  the iteration counter of the minimizer
            | ``f``  --  the function value at ``x_orig``
            | ``x_orig``  --  the unknowns in original coordinates
            | ``gradient_orig``  --  the gradient in original coordinates

           Return value:
            | ``do_update``  --  True when an update is required.

           Derived classes must call this method to test of the preconditioner
           requires updating. Derived classes must also return this boolean
           to their caller.
        """
        if counter - self.last_update > self.each:
            grad_rms = np.sqrt((gradient_orig**2).mean())
            if grad_rms < self.grad_rms:
                self.last_update = counter
                return True
        return False