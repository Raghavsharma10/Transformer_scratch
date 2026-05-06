def set_sample_coverage(self, value=1.0, invert=False):
        """Specify multisample coverage parameters
    
        Parameters
        ----------
        value : float
            Sample coverage value (will be clamped between 0. and 1.).
        invert : bool
            Specify if the coverage masks should be inverted.
        """
        self.glir.command('FUNC', 'glSampleCoverage', float(value), 
                          bool(invert))