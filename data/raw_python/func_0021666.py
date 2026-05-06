def refine(self):
        """
        Return a refined CSG. To each polygon, a middle point is added to each edge and to the center 
        of the polygon
        """
        newCSG = CSG()
        for poly in self.polygons:

            verts = poly.vertices
            numVerts = len(verts)

            if numVerts == 0:
                continue

            midPos = reduce(operator.add, [v.pos for v in verts]) / float(numVerts)
            midNormal = None
            if verts[0].normal is not None:
                midNormal = poly.plane.normal
            midVert = Vertex(midPos, midNormal)

            newVerts = verts + \
                       [verts[i].interpolate(verts[(i + 1)%numVerts], 0.5) for i in range(numVerts)] + \
                       [midVert]

            i = 0
            vs = [newVerts[i], newVerts[i+numVerts], newVerts[2*numVerts], newVerts[2*numVerts-1]]
            newPoly = Polygon(vs, poly.shared)
            newPoly.shared = poly.shared
            newPoly.plane = poly.plane
            newCSG.polygons.append(newPoly)

            for i in range(1, numVerts):
                vs = [newVerts[i], newVerts[numVerts+i], newVerts[2*numVerts], newVerts[numVerts+i-1]]
                newPoly = Polygon(vs, poly.shared)
                newCSG.polygons.append(newPoly)
                
        return newCSG