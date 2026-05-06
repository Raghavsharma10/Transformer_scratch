def is_touched(self, position):
        """Hit detection method.
        
        Indicates if this key has been hit by a touch / click event at the given position.

        :param position: Event position.
        :returns: True is the given position collide this key, False otherwise.
        """
        return position[0] >= self.position[0] and position[0] <= self.position[0]+ self.size[0]