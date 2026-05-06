def evaluate(self, x, *args):
        """Return flux in PHOTLAM. Assume input wavelength is in Angstrom."""
        xx = x / self.x_0
        y = (self.amplitude * xx ** (-self.alpha)) * self._flux_unit
        flux = units.convert_flux(x, y, units.PHOTLAM)
        return flux.value