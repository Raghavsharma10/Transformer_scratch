def inverse(self):
        """
        Return a new CSG solid with solid and empty space switched. This solid is
        not modified.
        """
        csg = self.clone()
        map(lambda p: p.flip(), csg.polygons)
        return csg