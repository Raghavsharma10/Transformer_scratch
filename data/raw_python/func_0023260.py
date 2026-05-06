def resize_bytes(self, size):
        """ Resize this buffer (deferred operation). 
        
        Parameters
        ----------
        size : int
            New buffer size in bytes.
        """
        self._nbytes = size
        self._glir.command('SIZE', self._id, size)
        # Invalidate any view on this buffer
        for view in self._views:
            if view() is not None:
                view()._valid = False
        self._views = []