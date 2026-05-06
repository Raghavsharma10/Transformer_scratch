def _sumindex(self, index=None):
        """Convert tuple index to 1-D index into value"""
        try:
            ndim = len(index)
        except TypeError:
            # turn index into a 1-tuple
            index = (index,)
            ndim = 1
        if len(self.shape) != ndim:
            raise ValueError("Index to %d-dimensional array %s has too %s dimensions" %
                (len(self.shape), self.name, ["many","few"][len(self.shape) > ndim]))
        sumindex = 0
        for i in range(ndim-1,-1,-1):
            index1 = index[i]
            if index1 < 0 or index1 >= self.shape[i]:
                raise ValueError("Dimension %d index for array %s is out of bounds (value=%d)" %
                    (i+1, self.name, index1))
            sumindex = index1 + sumindex*self.shape[i]
        return sumindex