def allPolygons(self):
        """
        Return a list of all polygons in this BSP tree.
        """
        polygons = self.polygons[:]
        if self.front: 
            polygons.extend(self.front.allPolygons())
        if self.back: 
            polygons.extend(self.back.allPolygons())
        return polygons