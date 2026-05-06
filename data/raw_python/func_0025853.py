def _data(self):
        """ Returns the data array associated with this file/extenstion."""
        hdu = self.open()
        _data = hdu.data.copy()
        if not self.inmemory:
            self.close()
            del hdu
        return _data