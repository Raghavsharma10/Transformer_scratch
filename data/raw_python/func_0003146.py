def get_SZ(self, psd, geometry):
        """
        Compute the scattering matrices for the given PSD and geometries.

        Returns:
            The new amplitude (S) and phase (Z) matrices.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError(
                "Initialize or load the scattering table first.")

        if (not isinstance(psd, PSD)) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = psd(self._psd_D)

            for geom in self.geometries:
                self._S_dict[geom] = \
                    trapz(self._S_table[geom] * psd_w, self._psd_D)
                self._Z_dict[geom] = \
                    trapz(self._Z_table[geom] * psd_w, self._psd_D)

            self._previous_psd = psd

        return (self._S_dict[geometry], self._Z_dict[geometry])