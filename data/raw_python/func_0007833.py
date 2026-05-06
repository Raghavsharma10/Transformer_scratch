def reMutualReceptions(self):
        """ Returns all mutual receptions with the object
        and other planets, indexed by planet ID. 
        It only includes ruler and exaltation receptions.
        
        """
        planets = copy(const.LIST_SEVEN_PLANETS)
        planets.remove(self.obj.id)
        mrs = {}
        for ID in planets:
            mr = self.dyn.reMutualReceptions(self.obj.id, ID)
            if mr:
                mrs[ID] = mr
        return mrs