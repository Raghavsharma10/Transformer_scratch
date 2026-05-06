def clipPolygons(self, polygons):
        """ 
        Recursively remove all polygons in `polygons` that are inside this BSP
        tree.
        """
        if not self.plane: 
            return polygons[:]

        front = []
        back = []
        for poly in polygons:
            self.plane.splitPolygon(poly, front, back, front, back)

        if self.front: 
            front = self.front.clipPolygons(front)

        if self.back: 
            back = self.back.clipPolygons(back)
        else:
            back = []

        front.extend(back)
        return front