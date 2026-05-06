def _reshape_n_vecs(self):
        """return list of arrays, each array represents a different m mode"""

        lst = []
        sl = slice(None, None, None)
        lst.append(self.__getitem__((sl, 0)))
        for m in xrange(1, self.mmax + 1):
            lst.append(self.__getitem__((sl, -m)))
            lst.append(self.__getitem__((sl, m)))
        return lst