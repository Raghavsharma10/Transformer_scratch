def _reshape_m_vecs(self):
        """return list of arrays, each array represents a different n mode"""
        
        lst = []
        for n in xrange(0, self.nmax + 1):
            mlst = []
            if n <= self.mmax:
                nn = n
            else:
                nn = self.mmax            
            for m in xrange(-nn, nn + 1):
                mlst.append(self.__getitem__((n, m)))
            lst.append(mlst)
        return lst