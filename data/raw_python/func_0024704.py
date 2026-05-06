def _process_flux_param(self, pval, wave):
        """Process individual model parameter representing flux."""
        if isinstance(pval, u.Quantity):
            self._validate_flux_unit(pval.unit)
            outval = units.convert_flux(self._redshift_model(wave), pval,
                                        self._internal_flux_unit).value
        else:  # Assume already in internal unit
            outval = pval
        return outval