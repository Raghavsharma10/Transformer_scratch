def movement(self):
        """ Returns if this object is direct, retrograde 
        or stationary. 
        
        """
        if abs(self.lonspeed) < 0.0003:
            return const.STATIONARY
        elif self.lonspeed > 0:
            return const.DIRECT
        else:
            return const.RETROGRADE