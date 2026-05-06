def splitPolygon(self, polygon, coplanarFront, coplanarBack, front, back):
        """
        Split `polygon` by this plane if needed, then put the polygon or polygon
        fragments in the appropriate lists. Coplanar polygons go into either
        `coplanarFront` or `coplanarBack` depending on their orientation with
        respect to this plane. Polygons in front or in back of this plane go into
        either `front` or `back`
        """
        COPLANAR = 0 # all the vertices are within EPSILON distance from plane
        FRONT = 1 # all the vertices are in front of the plane
        BACK = 2 # all the vertices are at the back of the plane
        SPANNING = 3 # some vertices are in front, some in the back

        # Classify each point as well as the entire polygon into one of the above
        # four classes.
        polygonType = 0
        vertexLocs = []
        
        numVertices = len(polygon.vertices)
        for i in range(numVertices):
            t = self.normal.dot(polygon.vertices[i].pos) - self.w
            loc = -1
            if t < -Plane.EPSILON: 
                loc = BACK
            elif t > Plane.EPSILON: 
                loc = FRONT
            else: 
                loc = COPLANAR
            polygonType |= loc
            vertexLocs.append(loc)
    
        # Put the polygon in the correct list, splitting it when necessary.
        if polygonType == COPLANAR:
            normalDotPlaneNormal = self.normal.dot(polygon.plane.normal)
            if normalDotPlaneNormal > 0:
                coplanarFront.append(polygon)
            else:
                coplanarBack.append(polygon)
        elif polygonType == FRONT:
            front.append(polygon)
        elif polygonType == BACK:
            back.append(polygon)
        elif polygonType == SPANNING:
            f = []
            b = []
            for i in range(numVertices):
                j = (i+1) % numVertices
                ti = vertexLocs[i]
                tj = vertexLocs[j]
                vi = polygon.vertices[i]
                vj = polygon.vertices[j]
                if ti != BACK: 
                    f.append(vi)
                if ti != FRONT:
                    if ti != BACK: 
                        b.append(vi.clone())
                    else:
                        b.append(vi)
                if (ti | tj) == SPANNING:
                    # interpolation weight at the intersection point
                    t = (self.w - self.normal.dot(vi.pos)) / self.normal.dot(vj.pos.minus(vi.pos))
                    # intersection point on the plane
                    v = vi.interpolate(vj, t)
                    f.append(v)
                    b.append(v.clone())
            if len(f) >= 3: 
                front.append(Polygon(f, polygon.shared))
            if len(b) >= 3: 
                back.append(Polygon(b, polygon.shared))