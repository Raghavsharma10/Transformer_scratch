def thermal_source(self):
        """Apply emissivity to an existing beam to produce a thermal
        source spectrum (without optical counterpart).

        Thermal source spectrum is calculated as follow:

            #. Create a blackbody spectrum in PHOTLAM per square arcsec
               with `temperature`.
            #. Multiply the blackbody with `beam_fill_factor` and ``self``.

        Returns
        -------
        sp : `~synphot.spectrum.SourceSpectrum`
            Thermal source spectrum.

        """
        sp = (SourceSpectrum(BlackBody1D, temperature=self.temperature) *
              units.SR_PER_ARCSEC2 * self.beam_fill_factor * self)
        sp.meta['temperature'] = self.temperature
        sp.meta['beam_fill_factor'] = self.beam_fill_factor
        return sp