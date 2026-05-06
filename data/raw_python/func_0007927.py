def movement(self):
        """ Returns the movement of this aspect. 
        The movement is the one of the active object, except
        if the active is separating but within less than 1 
        degree.
        
        """
        mov = self.active.movement
        if self.orb < 1 and mov == const.SEPARATIVE:
            mov = const.EXACT
        return mov