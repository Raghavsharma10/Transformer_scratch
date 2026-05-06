def save_uca(self, rootpath, raw=False, as_int=False):
        """ Saves the upstream contributing area to a file
        """
        self.save_array(self.uca, None, 'uca', rootpath, raw, as_int=as_int)