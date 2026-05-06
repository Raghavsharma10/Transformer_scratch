def do(self, x_orig):
        """Transform the unknowns to preconditioned coordinates

           This method also transforms the gradient to original coordinates
        """
        if self.scales is None:
            return x_orig
        else:
            return np.dot(self.rotation.transpose(), x_orig)*self.scales