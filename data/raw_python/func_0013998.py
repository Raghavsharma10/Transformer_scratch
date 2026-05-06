def merge(self, other):
        """Adds metadata variables to self that are in other but not in self.
        
        Parameters
        ----------
        other : pysat.Meta
        
        """
        
        for key in other.keys():
            if key not in self:
                # copies over both lower and higher dimensional data
                self[key] = other[key]