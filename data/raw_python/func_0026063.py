def set_pscale(self):
        """ Compute the pixel scale based on active WCS values. """
        if self.new:
            self.pscale = 1.0
        else:
            self.pscale = self.compute_pscale(self.cd11,self.cd21)