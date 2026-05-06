def isFeral(self):
        """ Returns true if the object does not have any 
        aspects.
        
        """
        planets = copy(const.LIST_SEVEN_PLANETS)
        planets.remove(self.obj.id)
        for otherID in planets:
            otherObj = self.chart.getObject(otherID)
            if aspects.hasAspect(self.obj, otherObj, const.MAJOR_ASPECTS):
                return False
        return True