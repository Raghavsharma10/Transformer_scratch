def getArc(self, prom, sig):
        """ Returns the arcs between a promissor and
        a significator. Should uses the object creation 
        functions to build the objects.
        
        """
        res = self._arc(prom, sig)
        res.update({
            'prom': prom['id'],
            'sig': sig['id']
        })
        return res