def _elements(self, IDs, func, aspList):
        """ Returns the IDs as objects considering the
        aspList and the function.
        
        """
        res = []
        for asp in aspList:
            if (asp in [0, 180]):
                # Generate func for conjunctions and oppositions
                if func == self.N:
                    res.extend([func(ID, asp) for ID in IDs])
                else:
                    res.extend([func(ID) for ID in IDs])
            else:
                # Generate Dexter and Sinister for others
                res.extend([self.D(ID, asp) for ID in IDs])
                res.extend([self.S(ID, asp) for ID in IDs])
        return res