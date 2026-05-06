def model(self):
        """Model of the spectrum with given redshift."""
        if self.z == 0:
            m = self._model
        else:
            # wavelength
            if self._internal_wave_unit.physical_type == 'length':
                rs = self._redshift_model.inverse
            # frequency or wavenumber
            # NOTE: This will never execute as long as internal wavelength
            #       unit remains Angstrom.
            else:  # pragma: no cover
                rs = self._redshift_model

            if self.z_type == 'wavelength_only':
                m = rs | self._model
            else:  # conserve_flux
                m = rs | self._model | self._redshift_flux_model

        return m