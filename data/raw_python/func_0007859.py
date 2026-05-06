def _buildTerms(self):
        """ Builds a data structure indexing the terms
        longitude by sign and object.
        
        """
        termLons = tables.termLons(tables.EGYPTIAN_TERMS)
        res = {}
        for (ID, sign, lon) in termLons:
            try:
                res[sign][ID] = lon
            except KeyError:
                res[sign] = {}
                res[sign][ID] = lon
        return res