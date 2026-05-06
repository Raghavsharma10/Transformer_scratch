def build(self, polygons):
        """
        Build a BSP tree out of `polygons`. When called on an existing tree, the
        new polygons are filtered down to the bottom of the tree and become new
        nodes there. Each set of polygons is partitioned using the first polygon
        (no heuristic is used to pick a good split).
        """
        if len(polygons) == 0:
            return
        if not self.plane: 
            self.plane = polygons[0].plane.clone()
        # add polygon to this node
        self.polygons.append(polygons[0])
        front = []
        back = []
        # split all other polygons using the first polygon's plane
        for poly in polygons[1:]:
            # coplanar front and back polygons go into self.polygons
            self.plane.splitPolygon(poly, self.polygons, self.polygons,
                                    front, back)
        # recursively build the BSP tree
        if len(front) > 0:
            if not self.front:
                self.front = BSPNode()
            self.front.build(front)
        if len(back) > 0:
            if not self.back:
                self.back = BSPNode()
            self.back.build(back)