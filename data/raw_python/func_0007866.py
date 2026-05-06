def _arc(self, prom, sig):
        """ Computes the in-zodiaco and in-mundo arcs 
        between a promissor and a significator.
        
        """
        arcm = arc(prom['ra'], prom['decl'], 
                   sig['ra'], sig['decl'], 
                   self.mcRA, self.lat)
        arcz = arc(prom['raZ'], prom['declZ'], 
                   sig['raZ'], sig['declZ'], 
                   self.mcRA, self.lat)
        return {
            'arcm': arcm,
            'arcz': arcz
        }