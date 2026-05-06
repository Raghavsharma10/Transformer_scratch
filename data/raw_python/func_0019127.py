def calc_secondary_parameters(self):
        """Determine the values of the secondary parameters `a` and `b`."""
        self.a = self.x/(2.*self.d**.5)
        self.b = self.u/(2.*self.d**.5)