def save_image(self, outname):
        """
        Save the image data.
        This is probably only useful if the image data has been blanked.

        Parameters
        ----------
        outname : str
            Name for the output file.
        """
        hdu = self.global_data.img.hdu
        hdu.data = self.global_data.img._pixels
        hdu.header["ORIGIN"] = "Aegean {0}-({1})".format(__version__, __date__)
        # delete some axes that we aren't going to need
        for c in ['CRPIX3', 'CRPIX4', 'CDELT3', 'CDELT4', 'CRVAL3', 'CRVAL4', 'CTYPE3', 'CTYPE4']:
            if c in hdu.header:
                del hdu.header[c]
        hdu.writeto(outname, overwrite=True)
        self.log.info("Wrote {0}".format(outname))
        return