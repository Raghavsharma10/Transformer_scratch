def aspects(self, obj):
        """ Returns true if this star aspects another object.
        Fixed stars only aspect by conjunctions. 
        
        """
        dist = angle.closestdistance(self.lon, obj.lon)
        return abs(dist) < self.orb()