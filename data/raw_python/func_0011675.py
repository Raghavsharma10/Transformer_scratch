def split(self, indices_or_sections, axis=0):
        """Split a timeseries into multiple sub-timeseries"""
        if not isinstance(indices_or_sections, numbers.Integral):
            raise Error('splitting by array of indices is not yet implemented')
        n = indices_or_sections
        if self.shape[axis] % n != 0:
            raise ValueError("Array split doesn't result in an equal division")
        step = self.shape[axis] / n
        pieces = []
        start = 0
        while start < self.shape[axis]:
            stop = start + step
            ix = [slice(None)] * self.ndim
            ix[axis] = slice(start, stop)
            ix = tuple(ix)
            pieces.append(self[ix])
            start += step
        return pieces