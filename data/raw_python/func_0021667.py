def translate(self, disp):
        """
        Translate Geometry.
           disp: displacement (array of floats)
        """
        d = Vector(disp[0], disp[1], disp[2])
        for poly in self.polygons:
            for v in poly.vertices:
                v.pos = v.pos.plus(d)