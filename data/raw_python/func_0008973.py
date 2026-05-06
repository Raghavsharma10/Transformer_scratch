def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)