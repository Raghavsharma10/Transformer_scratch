def save_direction(self, rootpath, raw=False, as_int=False):
        """ Saves the direction of the slope to a file
        """
        self.save_array(self.direction, None, 'ang', rootpath, raw, as_int=as_int)