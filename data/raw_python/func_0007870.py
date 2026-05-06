def getList(self, aspList):
        """ Returns a sorted list with all
        primary directions. 
        
        """
        # Significators
        objects = self._elements(self.SIG_OBJECTS, self.N, [0])
        houses = self._elements(self.SIG_HOUSES, self.N, [0])
        angles = self._elements(self.SIG_ANGLES, self.N, [0])
        significators = objects + houses + angles
        
        # Promissors
        objects = self._elements(self.SIG_OBJECTS, self.N, aspList)
        terms = self._terms()
        antiscias = self._elements(self.SIG_OBJECTS, self.A, [0])
        cantiscias = self._elements(self.SIG_OBJECTS, self.C, [0])
        promissors = objects + terms + antiscias + cantiscias

        # Compute all
        res = []
        for prom in promissors:
            for sig in significators:
                if (prom['id'] == sig['id']):
                    continue
                arcs = self._arc(prom, sig)
                for (x,y) in [('arcm', 'M'), ('arcz', 'Z')]:
                    arc = arcs[x]
                    if 0 < arc < self.MAX_ARC:
                        res.append([
                            arcs[x],
                            prom['id'],
                            sig['id'],
                            y,
                        ])

        return sorted(res)