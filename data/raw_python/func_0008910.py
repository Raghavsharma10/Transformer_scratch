def save_outputs(self, rootpath='.', raw=False):
        """Saves TWI, UCA, magnitude and direction of slope to files.
        """
        self.save_twi(rootpath, raw)
        self.save_uca(rootpath, raw)
        self.save_slope(rootpath, raw)
        self.save_direction(rootpath, raw)