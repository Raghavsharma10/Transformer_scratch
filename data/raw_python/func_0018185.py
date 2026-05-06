def write_corrected(self, output, clobber=False):
        """Write out the destriped data."""

        # un-apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science / self.invflat
            self.err = self.err / self.invflat

        # un-apply the post-flash if necessary
        if self.flshcorr != 'COMPLETE':
            self.science = self.science + self.flash

        # un-apply the dark if necessary
        if self.darkcorr != 'COMPLETE':
            self.science = self.science + self.dark

        # reverse the amp merge
        if (self.ampstring == 'ABCD'):
            tmp_1, tmp_2 = np.split(self.science, 2, axis=1)
            self.hdulist['sci', 1].data = tmp_1.copy()
            self.hdulist['sci', 2].data = tmp_2[::-1, :].copy()

            tmp_1, tmp_2 = np.split(self.err, 2, axis=1)
            self.hdulist['err', 1].data = tmp_1.copy()
            self.hdulist['err', 2].data = tmp_2[::-1, :].copy()
        else:
            self.hdulist['sci', 1].data = self.science.copy()
            self.hdulist['err', 1].data = self.err.copy()

        # Write the output
        self.hdulist.writeto(output, overwrite=clobber)