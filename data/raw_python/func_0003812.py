def results(self):
        """Return the value and optionally derivative and second order derivative"""
        if self.deriv == 0:
            return self.v,
        if self.deriv == 1:
            return self.v, self.d
        if self.deriv == 2:
            return self.v, self.d, self.dd