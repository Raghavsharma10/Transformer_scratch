def open(self):
        """ Opens the file for subsequent access. """

        if self.handle is None:
            self.handle = fits.open(self.fname, mode='readonly')

        if self.extn:
            if len(self.extn) == 1:
                hdu = self.handle[self.extn[0]]
            else:
                hdu = self.handle[self.extn[0],self.extn[1]]
        else:
            hdu = self.handle[0]
        if isinstance(hdu,fits.hdu.compressed.CompImageHDU):
            self.compress = True
        return hdu