def clipTo(self, bsp):
        """ 
        Remove all polygons in this BSP tree that are inside the other BSP tree
        `bsp`.
        """
        self.polygons = bsp.clipPolygons(self.polygons)
        if self.front: 
            self.front.clipTo(bsp)
        if self.back: 
            self.back.clipTo(bsp)