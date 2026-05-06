def rotate(self, axis, angleDeg):
        """
        Rotate geometry.
           axis: axis of rotation (array of floats)
           angleDeg: rotation angle in degrees
        """
        ax = Vector(axis[0], axis[1], axis[2]).unit()
        cosAngle = math.cos(math.pi * angleDeg / 180.)
        sinAngle = math.sin(math.pi * angleDeg / 180.)

        def newVector(v):
            vA = v.dot(ax)
            vPerp = v.minus(ax.times(vA))
            vPerpLen = vPerp.length()
            if vPerpLen == 0:
                # vector is parallel to axis, no need to rotate
                return v
            u1 = vPerp.unit()
            u2 = u1.cross(ax)
            vCosA = vPerpLen*cosAngle
            vSinA = vPerpLen*sinAngle
            return ax.times(vA).plus(u1.times(vCosA).plus(u2.times(vSinA)))

        for poly in self.polygons:
            for vert in poly.vertices:
                vert.pos = newVector(vert.pos)
                normal = vert.normal
                if normal.length() > 0:
                    vert.normal = newVector(vert.normal)