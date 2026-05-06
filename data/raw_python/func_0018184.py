def ingest_dark(self):
        """Process dark."""

        self.dark = extract_dark(self.hdulist[0].header, self.hdulist[1])

        # If BIAS or DARK, set dark to zeros
        if self.dark is None:
            self.dark = np.zeros_like(self.science)
            return

        # Apply the dark subtraction if necessary.
        # Effect of DARK on ERR is insignificant for de-striping.
        if self.darkcorr != 'COMPLETE':
            self.science = self.science - self.dark