def save_slope(self, rootpath, raw=False, as_int=False):
        """ Saves the magnitude of the slope to a file
        """
        self.save_array(self.mag, None, 'mag', rootpath, raw, as_int=as_int)