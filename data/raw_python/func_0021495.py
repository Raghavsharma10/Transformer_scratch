def getFields(self):
        """
        Returns all the class attributues.
        
        @rtype: dict
        @return: A dictionary containing all the class attributes.
        """
        d = {}
        for i in self._attrsList:
            key = i
            value = getattr(self,  i)
            d[key] = value
        return d