def invert(self):
        """ 
        Convert solid space to empty space and empty space to solid space.
        """
        for poly in self.polygons:
            poly.flip()
        self.plane.flip()
        if self.front: 
            self.front.invert()
        if self.back: 
            self.back.invert()
        temp = self.front
        self.front = self.back
        self.back = temp