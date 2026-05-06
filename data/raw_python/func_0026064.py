def compute_pscale(self,cd11,cd21):
        """ Compute the pixel scale based on active WCS values. """
        return N.sqrt(N.power(cd11,2)+N.power(cd21,2)) * 3600.