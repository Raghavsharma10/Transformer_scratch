def ingest_flatfield(self):
        """Process flatfield."""

        self.invflat = extract_flatfield(
            self.hdulist[0].header, self.hdulist[1])

        # If BIAS or DARK, set flatfield to unity
        if self.invflat is None:
            self.invflat = np.ones_like(self.science)
            return

        # Apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science * self.invflat
            self.err = self.err * self.invflat