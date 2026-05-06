def type(self):
        """ Returns the shape of the data array associated with this file."""
        hdu = self.open()
        _type = hdu.data.dtype.name
        if not self.inmemory:
            self.close()
            del hdu
        return _type