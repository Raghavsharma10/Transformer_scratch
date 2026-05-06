def _shape(self):
        """ Returns the shape of the data array associated with this file."""
        hdu = self.open()
        _shape = hdu.shape
        if not self.inmemory:
            self.close()
            del hdu
        return _shape