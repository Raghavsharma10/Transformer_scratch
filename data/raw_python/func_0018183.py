def ingest_flash(self):
        """Process post-flash."""

        self.flash = extract_flash(self.hdulist[0].header, self.hdulist[1])

        # Set post-flash to zeros
        if self.flash is None:
            self.flash = np.zeros_like(self.science)
            return

        # Apply the flash subtraction if necessary.
        # Not applied to ERR, to be consistent with ingest_dark()
        if self.flshcorr != 'COMPLETE':
            self.science = self.science - self.flash