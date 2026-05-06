def _init_bins(self, binset):
        """Calculated binned wavelength centers, edges, and flux.

        By contrast, the native waveset and flux should be considered
        samples of a continuous function.

        Thus, it makes sense to interpolate ``self.waveset`` and
        ``self(self.waveset)``, but not `binset` and `binflux`.

        """
        if binset is None:
            if self.bandpass.waveset is not None:
                self._binset = self.bandpass.waveset
            elif self.spectrum.waveset is not None:
                self._binset = self.spectrum.waveset
                log.info('Bandpass waveset is undefined; '
                         'Using source spectrum waveset instead.')
            else:
                raise exceptions.UndefinedBinset(
                    'Both source spectrum and bandpass have undefined '
                    'waveset; Provide binset manually.')
        else:
            self._binset = self._validate_wavelengths(binset)

        # binset must be in ascending order for calcbinflux()
        # to work properly.
        if self._binset[0] > self._binset[-1]:
            self._binset = self._binset[::-1]

        self._bin_edges = binning.calculate_bin_edges(self._binset)

        # Merge bin edges and centers in with the natural waveset
        spwave = utils.merge_wavelengths(
            self._bin_edges.value, self._binset.value)
        if self.waveset is not None:
            spwave = utils.merge_wavelengths(spwave, self.waveset.value)

        # Throw out invalid wavelengths after merging.
        spwave = spwave[spwave > 0]

        # Compute indices associated to each endpoint.
        indices = np.searchsorted(spwave, self._bin_edges.value)
        i_beg = indices[:-1]
        i_end = indices[1:]

        # Prepare integration variables.
        flux = self(spwave)
        avflux = (flux.value[1:] + flux.value[:-1]) * 0.5
        deltaw = spwave[1:] - spwave[:-1]

        # Sum over each bin.
        binflux, intwave = binning.calcbinflux(
            self._binset.size, i_beg, i_end, avflux, deltaw)

        self._binflux = binflux * flux.unit