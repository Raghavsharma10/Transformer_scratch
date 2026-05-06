def calc_secondary_parameters(self):
        """Determine the value of the secondary parameter `c`."""
        self.c = 1./(self.k*special.gamma(self.n))