def lambda_max(self):
        """Peak wavelength in Angstrom when the curve is expressed as
        power density."""
        return ((const.b_wien.value / self.temperature) * u.m).to(u.AA).value