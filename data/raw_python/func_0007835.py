def __aspectLists(self, IDs, aspList):
        """ Returns a list with the aspects that the object
        makes to the objects in IDs. It considers only
        conjunctions and other exact/applicative aspects
        if in aspList.
        
        """
        res = []
        
        for otherID in IDs:
            # Ignore same 
            if otherID == self.obj.id:
                continue
            
            # Get aspects to the other object
            otherObj = self.chart.getObject(otherID)
            asp = aspects.getAspect(self.obj, otherObj, aspList)
            
            if asp.type == const.NO_ASPECT:
                continue
            elif asp.type == const.CONJUNCTION:
                res.append(asp.type)
            else:
                # Only exact or applicative aspects
                movement = asp.movement()
                if movement in [const.EXACT, const.APPLICATIVE]:
                    res.append(asp.type)
        
        return res