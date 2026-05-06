def load_tpf(self):
        '''
        Loads the target pixel file.

        '''

        if not self.loaded:
            if self._data is not None:
                data = self._data
            else:
                data = self._mission.GetData(
                         self.ID, season=self.season,
                         cadence=self.cadence,
                         clobber=self.clobber_tpf,
                         aperture_name=self.aperture_name,
                         saturated_aperture_name=self.saturated_aperture_name,
                         max_pixels=self.max_pixels,
                         saturation_tolerance=self.saturation_tolerance,
                         get_hires=self.get_hires,
                         get_nearby=self.get_nearby)
                if data is None:
                    raise Exception("Unable to retrieve target data.")
            self.cadn = data.cadn
            self.time = data.time
            self.model = np.zeros_like(self.time)
            self.fpix = data.fpix
            self.fraw = np.sum(self.fpix, axis=1)
            self.fpix_err = data.fpix_err
            self.fraw_err = np.sqrt(np.sum(self.fpix_err ** 2, axis=1))
            self.nanmask = data.nanmask
            self.badmask = data.badmask
            self.transitmask = np.array([], dtype=int)
            self.outmask = np.array([], dtype=int)
            self.aperture = data.aperture
            self.aperture_name = data.aperture_name
            self.apertures = data.apertures
            self.quality = data.quality
            self.Xpos = data.Xpos
            self.Ypos = data.Ypos
            self.mag = data.mag
            self.pixel_images = data.pixel_images
            self.nearby = data.nearby
            self.hires = data.hires
            self.saturated = data.saturated
            self.meta = data.meta
            self.bkg = data.bkg

            # Update the last breakpoint to the correct value
            self.breakpoints[-1] = len(self.time) - 1

            # Get PLD normalization
            self.get_norm()

            self.loaded = True