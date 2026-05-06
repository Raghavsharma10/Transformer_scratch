def configure_arrays(self):
        """Get the SCI and ERR data."""
        self.science = self.hdulist['sci', 1].data
        self.err = self.hdulist['err', 1].data
        self.dq = self.hdulist['dq', 1].data
        if (self.ampstring == 'ABCD'):
            self.science = np.concatenate(
                (self.science, self.hdulist['sci', 2].data[::-1, :]), axis=1)
            self.err = np.concatenate(
                (self.err, self.hdulist['err', 2].data[::-1, :]), axis=1)
            self.dq = np.concatenate(
                (self.dq, self.hdulist['dq', 2].data[::-1, :]), axis=1)
        self.ingest_dark()
        self.ingest_flash()
        self.ingest_flatfield()