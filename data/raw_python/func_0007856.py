def getObjectsAspecting(self, point, aspList):
        """ Returns a list of objects aspecting a point 
        considering a list of possible aspects.
        
        """
        res = []
        for obj in self:
            if obj.isPlanet() and aspects.isAspecting(obj, point, aspList):
                res.append(obj)
        return ObjectList(res)