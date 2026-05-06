def undo(self, x_prec):
        """Transform the unknowns to original coordinates

           This method also transforms the gradient to preconditioned coordinates
        """
        if self.scales is None:
            return x_prec
        else:
            return np.dot(self.rotation, x_prec/self.scales)